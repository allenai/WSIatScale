import json
import numpy as np
import os

class SpecialTokens:
    """
    Stop words:
    ourselves hers between Between yourself Yourself but But again Again there There about About once Once during During out Out very Very having Having with With they They own Own an An be Be some Some for For do Do its Its yours Yours such Such into Into of Of most Most itself other Other off Off is Is am Am or Or who Who as As from From him Him each Each the The themselves until Until below Below are Are we We these These your Your his His through Through don Don nor Nor me Me were Were her Her more More himself Himself this This down Down should Should our Our their Their while While above Above both Both up Up to To ours had Had she She all All no No when When at At any Any before Before them Them same Same and And been Been have Have in In will Will on On does Does then Then that That because Because what What over Over why Why so So did Did not Not now Now under Under he He you You herself has Has just Just where Where too Too only Only myself which Which those Those after After few Few whom being Being if If theirs my My against Against by By doing Doing it It how How further Further was Was here Here than Than

    """
    def __init__(self, model_hf_path):
        if model_hf_path == 'bert-large-cased-whole-word-masking':
            full_stop_token, CLS, SEP = 119, 101, 102

            stop_words = {9655, 4364, 1206, 3847, 3739, 26379, 1133, 1252, 1254, 5630, 1175, 1247, 1164, 3517, 1517, 2857, 1219, 1507, 1149, 3929, 1304, 6424, 1515, 5823, 1114, 1556, 1152, 1220, 1319, 13432, 1126, 1760, 1129, 4108, 1199, 1789, 1111, 1370, 1202, 2091, 1157, 2098, 6762, 25901, 1216, 5723, 1154, 14000, 1104, 2096, 1211, 2082, 2111, 1168, 2189, 1228, 8060, 1110, 2181, 1821, 7277, 1137, 2926, 1150, 2627, 1112, 1249, 1121, 1622, 1140, 15619, 1296, 2994, 1103, 1109, 2310, 1235, 5226, 2071, 12219, 1132, 2372, 1195, 1284, 1292, 1636, 1240, 2353, 1117, 1230, 1194, 4737, 1274, 1790, 4040, 16162, 1143, 2508, 1127, 8640, 1123, 1430, 1167, 3046, 1471, 20848, 1142, 1188, 1205, 5245, 1431, 9743, 1412, 3458, 1147, 2397, 1229, 1799, 1807, 12855, 1241, 2695, 1146, 3725, 1106, 1706, 17079, 1125, 6467, 1131, 1153, 1155, 1398, 1185, 1302, 1165, 1332, 1120, 1335, 1251, 6291, 1196, 2577, 1172, 23420, 1269, 14060, 1105, 1262, 1151, 18511, 1138, 4373, 1107, 1130, 1209, 3100, 1113, 1212, 1674, 7187, 1173, 1599, 1115, 1337, 1272, 2279, 1184, 1327, 1166, 3278, 1725, 2009, 1177, 1573, 1225, 2966, 1136, 1753, 1208, 1986, 1223, 2831, 1119, 1124, 1128, 1192, 1941, 1144, 10736, 1198, 2066, 1187, 2777, 1315, 6466, 1178, 2809, 1991, 1134, 5979, 1343, 4435, 1170, 1258, 1374, 17751, 2292, 1217, 6819, 1191, 1409, 19201, 1139, 1422, 1222, 8801, 1118, 1650, 1833, 27691, 1122, 1135, 1293, 1731, 1748, 6940, 1108, 3982, 1303, 3446, 1190, 16062}
            stop_words = stop_words.union(set((22755, 1232, 23567, 20262))) #Do not appear in Wikipedia.
            single_letters_and_punctuation = set(range(1103))

        elif model_hf_path == 'bert-large-uncased':
            full_stop_token, CLS, SEP = 1012, 101, 102
            full_stop_token = None # Don't want to break.

            stop_words = {2049, 9731, 2053, 2054, 2055, 2057, 2058, 2059, 2572, 2061, 2060, 2062, 2064, 2065, 2068, 2069, 2070, 2582, 2073, 2074, 2076, 2077, 2079, 2083, 2084, 2085, 2087, 2090, 2091, 2096, 2097, 2104, 2106, 2107, 2108, 2114, 2115, 2119, 2122, 2123, 2125, 2127, 6737, 2129, 2138, 2151, 2153, 3183, 2168, 2169, 2682, 2182, 3209, 2200, 2205, 2725, 2216, 2219, 2256, 2261, 17156, 2320, 2323, 2841, 2339, 14635, 2870, 2370, 4426, 2383, 2917, 4496, 2993, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2515, 2003, 2005, 2004, 2007, 2006, 2009, 2010, 2011, 2008, 2013, 2012, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2029, 2030, 2031, 2032, 2033, 5106, 2035, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046}
            single_letters_and_punctuation = set(range(1996))
            # single_letters_and_punctuation.add(2133) # adding ...
        else:
            raise NotImplementedError

        self.full_stop_token, self.CLS, self.SEP = full_stop_token, CLS, SEP
        self.stop_words_and_punctuation = stop_words.union(single_letters_and_punctuation)
        half_words_list_path = f"non-full-words/non-full-words-{model_hf_path}.npy"
        self.half_words_list = np.load(half_words_list_path) if half_words_list_path else None

        lemmatized_vocab_path = f"lemmatized_vocabs/lemmatized_vocabs-{model_hf_path}.json"
        self.lemmatized_vocab = {int(k): v for k, v in json.load(open(lemmatized_vocab_path, 'r')).items()}
        
    def valid_token(self, token):
        if token in [self.full_stop_token, self.CLS, self.SEP]:
            return False
        if token in self.stop_words_and_punctuation:
            return False
        if token in self.half_words_list:
            return False
        return True

    def full_words_tokens(self, tokenizer):
        vocab = tokenizer.get_vocab()
        ret = set([token for _, token in vocab.items() if self.valid_token(token)])
        return ret

    def lemmatize(self, token):
        # Should assume token in self.lemmatized_vocab
        return self.lemmatized_vocab[token]

    def tokens_to_annotate(self):
        return set([v for v in self.lemmatized_vocab.values() if self.valid_token(v)])