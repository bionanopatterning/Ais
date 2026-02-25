from chimerax.core.commands import run

paths = ['Z:/compu_projects/easymode/datasets/050_MHSP/segmented\\20220718_TS_043_10.00Apx__cytoplasmic_granule.mrc', 'Z:/compu_projects/easymode/datasets/050_MHSP/segmented\\20220718_TS_043_10.00Apx__membrane.mrc', 'Z:/compu_projects/easymode/datasets/050_MHSP/segmented\\20220718_TS_043_10.00Apx__microtubule.mrc', 'Z:/compu_projects/easymode/datasets/050_MHSP/segmented\\20220718_TS_043_10.00Apx__mitochondrion.mrc', 'Z:/compu_projects/easymode/datasets/050_MHSP/segmented\\20220718_TS_043_10.00Apx__ribosome.mrc', 'Z:/compu_projects/easymode/datasets/050_MHSP/segmented\\20220718_TS_043_10.00Apx__vimentin.mrc']
level = [97, 81, 89, 93, 85, 96]
colour = [(1.0, 0.40784314274787903, 0.0), (0.6823529601097107, 0.0, 1.0), (0.08235294371843338, 0.0, 1.0), (0.0, 0.5333333611488342, 1.0431373119354248), (1.0, 0.05098039284348488, 0.0), (1.0, 0.9529411792755127, 0.0)]
dust = [2917.05419921875, 1.0, 1.0, 1000000.0, 1.0, 1.0]
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