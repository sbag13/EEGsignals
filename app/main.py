from EdfFile import EdfFile

edf_file = EdfFile("../SC4001E0-PSG.edf")
e = edf_file.signals_list[1].getEpochs()
e[0].extractFeatures()
# print(edf_file.annotations)
    
# for sig in edf_file.signals_list:
#     print(sig.label)