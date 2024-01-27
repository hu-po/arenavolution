import yaml
import glob
import os

import matplotlib.pyplot as plt

results_dir = '/home/oop/dev/data/einvol.f84b79/ckpt'
plot_filepath = os.path.join(results_dir, 'test_accuracy_plot.png')

yaml_files = glob.glob(f'{results_dir}/results.r*.yaml')
rounds = []
test_acc = []
for file in yaml_files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    round_number = int(file.split('.')[-2].split('r')[-1])
    for key in data:
        rounds.append(round_number)
        test_acc.append(data[key]['test_accuracy'])

plt.scatter(rounds, test_acc)
plt.xlabel('round')
plt.ylabel('acc')
plt.title('Test Accuracy per Round')
plt.xlim(0, 32)
plt.ylim(0, 1)
plt.savefig(plot_filepath)
plt.show()
