from chimerax.core.commands import run

paths = ['Z:/em/HeLa_MPA_merged/segmented\\l16p3_6_10.00Apx__Ferritin.mrc', 'Z:/em/HeLa_MPA_merged/segmented\\l16p3_6_10.00Apx__IMPDH.mrc', 'Z:/em/HeLa_MPA_merged/segmented\\l16p3_6_10.00Apx__microtubule.mrc', 'Z:/em/HeLa_MPA_merged/segmented\\l16p3_6_10.00Apx__prohibitin.mrc', 'Z:/em/HeLa_MPA_merged/segmented\\l16p3_6_10.00Apx__ribosome.mrc', 'Z:/em/HeLa_MPA_merged/segmented\\l16p3_6_10.00Apx__tric.mrc']
level = [92, 104, 106, 105, 83, 95]
colour = [(0.25882354378700256, 0.8392156958580017, 0.6431372761726379), (1.0431373119354248, 0.9510958194732666, 0.0), (0.08235294371843338, 0.0, 1.0), (1.0, 0.9529411792755127, 0.0), (1.0, 0.05098039284348488, 0.0), (0.0, 0.5333333611488342, 1.0431373119354248)]
dust = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
bgclr = (0.9399999976158142, 0.9399999976158142, 0.9399999976158142)

for i in range(len(paths)):
    run(session, f'open "{paths[i]}"')
    run(session, f'volume #{i+1} level {level[i]}')
    run(session, f'color #{i+1} rgb({colour[i][0]},{colour[i][1]},{colour[i][2]})')
    run(session, f'surface dust #{i+1} size {dust[i]} metric volume')

run(session, f'set bgColor rgb({bgclr[0]},{bgclr[1]},{bgclr[2]})')
run(session, f'graphics silhouettes true')
run(session, f'lighting soft')
run(session, f'lighting shadows false')