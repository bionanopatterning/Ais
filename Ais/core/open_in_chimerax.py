from chimerax.core.commands import run

# These five data lines are placeholders; Ais fills them in at launch time (see
# SegmentationEditor._write_launch_script) and runs a temporary copy of this file.
paths = []
level = []
colour = []
dust = []
bgclr = [0.94, 0.94, 0.94]

for i in range(len(paths)):
    run(session, f'open "{paths[i]}"')
    run(session, f'volume #{i+1} level {level[i]}')
    run(session, f'color #{i+1} rgb({colour[i][0]},{colour[i][1]},{colour[i][2]})')
    run(session, f'surface dust #{i+1} size {dust[i]} metric volume')

run(session, f'set bgColor rgb({bgclr[0]},{bgclr[1]},{bgclr[2]})')
run(session, f'graphics silhouettes true')
run(session, f'lighting soft')
run(session, f'lighting shadows false')