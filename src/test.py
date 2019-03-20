import progressbar
from time import sleep

n_epochs = 100

epoch_bar = progressbar.ProgressBar(maxval=n_epochs, \
    widgets=[progressbar.Bar('=', '[', ']'), '', progressbar.Percentage()])

print("Process status")
epoch_bar.start()


for epoch in range(n_epochs):
    epoch_bar.update(epoch)
    sleep(0.1)
epoch_bar.finish()