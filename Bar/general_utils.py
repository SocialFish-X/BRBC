import time
import sys
import logging
import numpy as np
import tensorflow as tf

def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)


# 定义一个hook，用于显示训练进度
class ProgressHook(tf.train.SessionRunHook):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0

    def before_run(self, run_context):
        self.current_step += 1
        return tf.train.SessionRunArgs(fetches=[])

    def after_run(self, run_context, run_values):
        progress = float(self.current_step) / self.total_steps
        num_bars = int(progress * 40)
        num_spaces = 40 - num_bars
        bar = '#' * num_bars + ' ' * num_spaces
        print(f"\rTraining progress: [{bar}] {int(progress * 100)}%", end='')

    def end(self, session):
        print("\nTraining complete.")

# 定义一个hook，用于显示Loss变化
class LossSummaryHook(tf.train.SessionRunHook):
    def __init__(self, summary_dir, loss_tensor_name):
        self.summary_dir = summary_dir
        self.loss_tensor_name = loss_tensor_name  # 传递损失张量的名称

    def begin(self):
        self.summary_writer = tf.summary.FileWriterCache.get(self.summary_dir)

    def before_run(self, run_context):
        # 获取损失张量的引用
        self.loss_tensor = tf.get_default_graph().get_tensor_by_name(self.loss_tensor_name)
        return tf.train.SessionRunArgs(fetches=self.loss_tensor)

    def after_run(self, run_context, run_values):
        loss_value = run_values.results
        summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss_value)])
        global_step = run_context.session.run(tf.train.get_global_step())
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()

    def end(self, session):
        self.summary_writer.close()