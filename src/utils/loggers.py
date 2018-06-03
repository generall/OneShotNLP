from torchlite.torch.train_callbacks import TrainCallback


class ModelParamsLogger(TrainCallback):
    """
    This logger should extract specific params form model and save it to logs
    """

    def on_epoch_end(self, epoch, logs=None):
        for k, m in logs['models'].items():
            if hasattr(m, 'params'):
                epoch_logs = logs.get('metrics_logs', {})
                epoch_logs = {
                    **epoch_logs,
                    **m.params
                }
                logs['metrics_logs'] = epoch_logs

