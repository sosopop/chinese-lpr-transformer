package com.example.licenseplaterecognition

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.licenseplaterecognition.ui.theme.LicensePlateRecognitionTheme
import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.ui.platform.LocalContext
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.LongBuffer

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            LicensePlateRecognitionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    MainContent(
                        modifier = Modifier.padding(innerPadding),
                        context = this
                    )
                }
            }
        }
    }
}

@Composable
fun MainContent(modifier: Modifier = Modifier, context: Context) {
    var showSnackbar by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }

    Box(
        modifier = modifier.fillMaxSize().padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Greeting(name = "Android")
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = { showSnackbar = true }) {
                Text(text = "Test Button")
            }
        }

        if (showSnackbar) {
            LaunchedEffect(snackbarHostState) {
                val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
                sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
                val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
                val imageEncoderSession: OrtSession = ortEnv.createSession(readModel(context, "image_encoder.onnx"), sessionOptions)
                val textDecoderSession: OrtSession = ortEnv.createSession(readModel(context, "text_decoder.onnx"), sessionOptions)

                // 从assets中读取图像并预处理
                val image = preprocessImageFromAssets(context, "1.jpg")
                val memory = runImageEncoderInference(imageEncoderSession, ortEnv, image)
                val result = runTextDecoderInference(textDecoderSession, ortEnv, memory)

                imageEncoderSession.close()
                textDecoderSession.close()
                ortEnv.close()

                snackbarHostState.showSnackbar(result)
                showSnackbar = false
            }
        }

        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier.align(Alignment.BottomCenter)
        )
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    LicensePlateRecognitionTheme {
        MainContent(context = LocalContext.current)
    }
}

fun readModel(context: Context, modelName: String): ByteArray {
    try {
        context.assets.open(modelName).use { inputStream ->
            val byteArray = ByteArray(inputStream.available())
            inputStream.read(byteArray)
            return byteArray
        }
    } catch (e: IOException) {
        throw RuntimeException("Error reading model from assets", e)
    }
}

fun preprocessImageFromAssets(context: Context, assetName: String): FloatBuffer {
    val bitmap = context.assets.open(assetName).use { inputStream ->
        BitmapFactory.decodeStream(inputStream)
    }
    val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

    val floatBuffer = FloatBuffer.allocate(3 * 224 * 224)
    val pixels = IntArray(224 * 224)
    scaledBitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)

    for (i in pixels.indices) {
        val pixel = pixels[i]
        val r = ((pixel shr 16) and 0xFF) / 255.0f
        val g = ((pixel shr 8) and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f

        val rIndex = i
        val gIndex = i + 224 * 224
        val bIndex = i + 2 * 224 * 224

        floatBuffer.put(rIndex, r)
        floatBuffer.put(gIndex, g)
        floatBuffer.put(bIndex, b)
    }
    return floatBuffer
}

fun runImageEncoderInference(session: OrtSession, ortEnv: OrtEnvironment, inputBuffer: FloatBuffer): OnnxTensor {
    val inputName = session.inputNames.iterator().next()
    val shape = longArrayOf(1, 3, 224, 224)
    val tensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)

    val results = session.run(mapOf(inputName to tensor))
    val outputName = session.outputNames.iterator().next()
    val resultTensor = results[0] as OnnxTensor
    val resultArray = resultTensor.floatBuffer.array()

    return resultTensor
}

fun runTextDecoderInference(session: OrtSession, ortEnv: OrtEnvironment, memory: OnnxTensor): String {
    val vocabList = listOf(
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑'
    ).map { it.toString() }

    val vocab = LicensePlateVocab(vocabList)

    val maxLength = 16

    var expression = ""
    var generatedTokens = vocab.textToSequence(expression, maxLength, true, false, true).map { it.toLong() }

    while (true) {
        val tgtInput = LongBuffer.allocate(maxLength)
        tgtInput.put(generatedTokens.toLongArray())
        tgtInput.rewind()
        val ortInputs = mapOf("memory" to memory, "tgt" to OnnxTensor.createTensor(ortEnv, tgtInput, longArrayOf(1, maxLength.toLong())))
        val results = session.run(ortInputs)
        val outputTensor = results[0] as OnnxTensor
        val outputArray = outputTensor.floatBuffer.array()
        val outputShape = outputTensor.info.shape
        outputTensor.close()

        val currentSeqLength = expression.length

        val beginIdx = currentSeqLength * outputShape[2].toInt()
        val endIdx = (currentSeqLength + 1) * outputShape[2].toInt()
        val outputSlice = outputArray.sliceArray(beginIdx until endIdx)

        // Find the index of the max value along the last dimension (71) at the current sequence length (expression length)
        val nextToken = outputSlice.indices.maxByOrNull { outputSlice[it] } ?: vocab.eosIdx

        if (nextToken == vocab.eosIdx || currentSeqLength >= maxLength) {
            break
        }
        expression += vocab.sequenceToText(listOf(nextToken))
        generatedTokens = vocab.textToSequence(expression, maxLength,  true,  false, true).map { it.toLong() }
    }

    return expression
}