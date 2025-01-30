from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()


    # Set your local paths here.
    settings.davis_dir = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.youtubevos_dir = ''

    settings.network_path = r'E:/SOT/SUTrack-main/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = 'E:/SOT/SUTrack-main'
    settings.result_plot_path = r'E:/SOT/SUTrack-main/test/result_plots'
    settings.results_path = r'E:/SOT/SUTrack-main/test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'E:/SOT/SUTrack-main'
    settings.segmentation_path = r'E:/SOT/SUTrack-main/test/segmentation_results'

    # settings.network_path = fr'/home/xuxianda/SOT/SUTrack-main/test/networks'    # Where tracking networks are stored.
    # settings.prj_dir = fr'/home/xuxianda/SOT/SUTrack-main'
    # settings.result_plot_path = fr'/home/xuxianda/SOT/SUTrack-main/test/result_plots'
    # settings.results_path = fr'/home/xuxianda/SOT/SUTrack-main/test/tracking_results'    # Where to store tracking results
    # settings.save_dir = fr'/home/xuxianda/SOT/SUTrack-main'
    # settings.segmentation_path = fr'/home/xuxianda/SOT/SUTrack-main/test/segmentation_results'

    return settings

