from SVT.tester.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.itb_path = r'E:\SOT\SVT\data\itb'
    settings.lasot_extension_subset_path = r'E:\SOT\SVT\data\lasot_extension_subset'
    settings.lasot_lmdb_path = r'E:\SOT\SVT\data\lasot_lmdb'
    settings.lasot_path = r'E:\lasot'
    settings.network_path = r'E:\SOT\SVT\output\test/networks'    # Where tracking networks are stored.
    settings.nfs_path = r'E:\SOT\SVT\tracking\data\nfs'
    settings.prj_dir =r'E:\SOT\SVT\tracking'
    settings.result_plot_path = r'E:\SOT\SVT\tracking\output\test/result_plots'
    settings.results_path = r'E:\SOT\SVT\tracking\output\test/tracking_results'    # Where to store tracking results
    settings.save_dir = r'E:\SOT\SVT\tracking\output'
    settings.segmentation_path = r'E:\SOT\SVT\output\test/segmentation_results'

    settings.tn_packed_results_path = ''
    settings.tpl_path = ''

    return settings

