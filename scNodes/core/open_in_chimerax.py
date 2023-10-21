from chimerax.core.commands import run

paths = ['C:\\Users\\mart_\\Desktop\\Wolff_2020_SciMag\\g70502_volb4_rotx_Double membrane.mrc', 'C:\\Users\\mart_\\Desktop\\Wolff_2020_SciMag\\g70502_volb4_rotx_Intermediate filament.mrc', 'C:\\Users\\mart_\\Desktop\\Wolff_2020_SciMag\\g70502_volb4_rotx_Pore.mrc', 'C:\\Users\\mart_\\Desktop\\Wolff_2020_SciMag\\g70502_volb4_rotx_Ribosomes.mrc', 'C:\\Users\\mart_\\Desktop\\Wolff_2020_SciMag\\g70502_volb4_rotx_Single membrane.mrc', 'C:\\Users\\mart_\\Desktop\\Wolff_2020_SciMag\\g70502_volb4_rotx_Tubulin.mrc']
level = [41, 131, 80, 79, 84, 85]
colour = [(0.25882354378700256, 0.8392156958580017, 0.6431372761726379), (1.0, 0.9529411792755127, 0.0), (1.0, 0.40784314274787903, 0.0), (1.0, 0.05098039284348488, 0.0), (0.6823529601097107, 0.0, 1.0), (0.08235294371843338, 0.0, 1.0)]
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