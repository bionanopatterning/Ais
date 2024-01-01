from Ais.core.segmentation_editor import QueuedExtract

q = QueuedExtract('C:/Users/mart_/Desktop/test/g11001_volb4_rotx__Ribosomes.mrc', 128, 1.0, 10.0, 'C:/Users/mart_/Desktop/test', binning=4)

q.start()

while q.process.progress < 1.0:
    pass