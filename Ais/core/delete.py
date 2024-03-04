# import os
# import glob
#
# tifs = glob.glob("Z:/mgflast/20240301_2102305_SerialEM-TOMO_WN_Mart/Movies/*.tif")
# print(tifs)
# atifs = dict()
# for t in tifs:
#     angle = t.split("Movies\\")[-1].split("_")[1]
#     atifs[angle] = t
#
#
# out_lines = list()
#
# with open("Z:/mgflast/20240301_2102305_SerialEM-TOMO_WN_Mart/Movies/tomo003.mrc.mdoc", 'r') as f:
#     lines = f.readlines()
#     for l in lines:
#         if "SubFramePath = " in l:
#             angle = l.split("Movies\\")[-1].split("_")[1]
#             lpath = atifs[angle].split("\\")[-1]
#             out_lines.append("SubFramePath = X:\\WarpFolder2\\20240301_2102305_SerialEM-TOMO_WN_Mart\\Movies\\"+lpath)
#         else:
#             out_lines.append(l)
#
# with open("Z:/mgflast/20240301_2102305_SerialEM-TOMO_WN_Mart/Movies/tomo003_corrected.mdoc", 'w') as f:
#     f.writelines(out_lines)
#
