package com.example.licenseplaterecognition

class LicensePlateVocab(
    vocabList: List<String>,
    padToken: String = "#",
    eosToken: String = "$",
    bosToken: String = "^"
) {
    val vocabList: List<String> = vocabList + padToken + eosToken + bosToken
    val padToken: String = padToken
    val eosToken: String = eosToken
    val bosToken: String = bosToken
    val vocabDict: Map<String, Int> = this.vocabList.withIndex().associate { it.value to it.index }
    val idxDict: Map<Int, String> = this.vocabList.withIndex().associate { it.index to it.value }
    val padIdx: Int = vocabDict[padToken] ?: error("Pad token not found in vocab list")
    val eosIdx: Int = vocabDict[eosToken] ?: error("EOS token not found in vocab list")
    val bosIdx: Int = vocabDict[bosToken] ?: error("BOS token not found in vocab list")
    val vocabSize: Int = this.vocabList.size

    fun textToSequence(
        text: String,
        maxLength: Int,
        padToMaxLength: Boolean = true,
        addEos: Boolean = true,
        addBos: Boolean = true
    ): List<Int> {
        val sequence = mutableListOf<Int>()
        if (addBos) {
            sequence.add(bosIdx)  // Add BOS token at the beginning
        }
        for (char in text) {
            val charStr = char.toString()
            if (vocabDict.containsKey(charStr)) {
                sequence.add(vocabDict[charStr]!!)
            }
        }
        if (addEos) {
            sequence.add(eosIdx)  // Add EOS token at the end
        }
        if (sequence.size < maxLength) {
            if (padToMaxLength) {
                sequence.addAll(List(maxLength - sequence.size) { padIdx })
            }
        } else {
            return sequence.subList(0, maxLength)
        }
        return sequence
    }

    fun sequenceToText(sequence: List<Int>): String {
        return sequence.filter { it != padIdx && it != eosIdx && it != bosIdx }
            .joinToString("") { idxDict[it] ?: "" }
    }
}