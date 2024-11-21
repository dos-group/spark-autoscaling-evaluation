import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'serif'
pd.set_option('display.float_format', '{:,.3f}'.format)  # Disable scientific notation for large numbers
pd.set_option('display.width', 230)
pd.set_option('display.max_columns', 12)  # Display up to 10 columns before truncating


SERVERLESS_DATA_DIR = './data/serverless_dataproc_ls/'
REGULAR_DATA_FILE = './data/regular_dataproc/regular_datproc_ls.csv'

# Contemporary GCP N2 VM pricing data in USD
HOURLY_VCPU_COST = 0.040730
HOURLY_GB_MEM_COST = 0.005458

# Contemporary GCP Dataproc Serverless pricing model (only relative values)
HOURLY_VCPU_COST = 6
HOURLY_GB_MEM_COST = 1

# === Utility functions

def attach_column(df, column_name, value=None, values=None):
    new_column = pd.Series(len(df)*(value,)) if value else pd.Series(values)
    df.loc[df.index, column_name] = new_column

def get_rows(df, **kwargs):
    rows = df.query(" and ".join(f"{k} == {repr(v)}" for k, v in kwargs.items()))
    return rows

def get_row(df, **kwargs):
    rows = df.query(" and ".join(f"{k} == {repr(v)}" for k, v in kwargs.items()))
    assert len(rows) == 1, f"{rows}\n{kwargs}"
    return rows.iloc[0]

algorithm_names = {
        'grep': 'Grep',
        'groupbycount': 'GroupByCount',
        'join': 'Join',
        'kmeans': 'K-Means',
        'linearregression': 'Linear Regression',
        'logisticregression': 'Logistic Regression',
        'selectwhereorderby': 'SelectWhereOrderBy',
        'sort': 'Sort',
        'wordcount': 'Wordcount',
}


# === Loading data


def load_serverless_dataproc_data(data_dir):
    """
    Returns
    - A dictionary containing dataframes for the resource allocation over time
      for each gcp dataproc serverless experiment
    - A DataFrame with the runtimes resource usage
      for each gcp dataproc serverless experiment
    """

    CPU_CORES_PER_EXECUTOR = 4  # GCP dataproc serverless v2.1 default settings
    MEMORY_GB_PER_EXECUTOR = 16  # GCP dataproc serverless v2.1 default settings

    runtimes = []
    executor_counts = dict()

    for file_name in sorted(os.listdir(data_dir)):
        job_id = file_name.split('.')[0]

        df = pd.read_csv(data_dir+file_name)  # columns 'timestamp', 'executor_count'
        df['timestamp'] = df['timestamp'] / 1000  # milliseconds -> seconds

        df['cpu_cores_per_node'] = CPU_CORES_PER_EXECUTOR
        df['memory_gb_per_node'] = MEMORY_GB_PER_EXECUTOR

        # Get executor_counts
        df['total_cluster_memory_gb'] = \
            df.apply(lambda row: row['executor_count']*MEMORY_GB_PER_EXECUTOR, axis=1).astype(int)
        df['total_cluster_cpu_cores'] = \
            df.apply(lambda row: row['executor_count']*CPU_CORES_PER_EXECUTOR, axis=1).astype(int)
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min())
        executor_counts[job_id] = df

        # Get runtimes
        algorithm, ds_size, min_executors, max_executors = job_id.split('-')
        runtime = (df['timestamp'].max() - df['timestamp'].min())

        # Get resource usage
        cpu_core_seconds, memory_gb_seconds, executor_seconds = 0, 0, 0
        for old_allocation, new_allocation in zip(df[:-1].iloc(), df[1:].iloc()):
            duration = new_allocation['timestamp'] - old_allocation['timestamp']
            executor_seconds += old_allocation['executor_count'] * duration
            cpu_core_seconds += old_allocation['total_cluster_cpu_cores'] * duration
            memory_gb_seconds += old_allocation['total_cluster_memory_gb'] * duration
        runtimes.append((job_id, algorithm, ds_size, int(min_executors), int(max_executors),
                         runtime, round(cpu_core_seconds,2), round(memory_gb_seconds,2), executor_seconds))

    cols = ('job_id', 'algorithm', 'dataset_size', 'min_executors', 'max_executors',
            'runtime', 'cpu_core_seconds', 'memory_gb_seconds', 'executor_seconds')
    runtimes_df = pd.DataFrame(runtimes, columns=cols)


    dynamic_only = (runtimes_df['min_executors'] == 2)
    static_only = (runtimes_df['min_executors'] != 2)
    runtimes_df.loc[static_only, 'mean_scaleout'] = \
            runtimes_df[static_only]['min_executors']
    runtimes_df.loc[dynamic_only, 'mean_scaleout'] = \
            runtimes_df[dynamic_only]['executor_seconds'] / \
            runtimes_df[dynamic_only]['runtime']
    runtimes_df['mean_total_cluster_cpu_cores'] = \
            (runtimes_df['mean_scaleout'] * CPU_CORES_PER_EXECUTOR).astype(int)
    runtimes_df['mean_total_cluster_memory_gb'] = \
            (runtimes_df['mean_scaleout'] * MEMORY_GB_PER_EXECUTOR).astype(int)

    def get_config_name(row):
        return ('S1', 'S2', 'S3')[int(row['min_executors']/8)]

    runtimes_df['config_name'] = runtimes_df.apply(get_config_name, axis=1)

    return runtimes_df, executor_counts


def load_regular_dataproc_data(data_file):

    config_names = {
        (8, 64, 64):    'R01',
        (8, 64, 256):   'R02',
        (8, 64, 512):   'R03',
        (4, 16, 128):   'R04',
        (4, 32, 128):   'R05',
        (4, 128, 128):  'R06',
        (2, 16, 128):   'R07',
        (8, 32, 128):   'R08',
        (16, 64, 256):  'R09',
        (16, 128, 128): 'R10',
    }
    def get_config_name(row):
        return config_names[(
            row['scaleout'],
            row['total_cluster_cpu_cores'],
            row['total_cluster_memory_gb'],
        )]

    cloud_df = pd.read_csv(data_file)

    # Derive additional values of interest
    cloud_df['total_cluster_memory_gb'] = cloud_df['memory_gb_per_node'] * cloud_df['scaleout']
    cloud_df['total_cluster_cpu_cores'] = cloud_df['cpu_cores_per_node'] * cloud_df['scaleout']
    cloud_df['memory_gb_per_cpu_core'] = cloud_df['memory_gb_per_node'] // cloud_df['cpu_cores_per_node']
    cloud_df['memory_gb_seconds'] = cloud_df['total_cluster_memory_gb'] * cloud_df['runtime']
    cloud_df['cpu_core_seconds'] = cloud_df['total_cluster_cpu_cores'] * cloud_df['runtime']
    cloud_df['dollar_cost'] = \
        cloud_df['cpu_core_seconds'] * HOURLY_VCPU_COST/3600 \
        + cloud_df['memory_gb_seconds'] * HOURLY_GB_MEM_COST/3600


    cloud_df['config_name'] = cloud_df.apply(get_config_name, axis=1)
    return cloud_df


# === Inferring additional dataset columns


def with_dollar_cost(df, hourly_vcpu_cost, hourly_gb_mem_cost):
    df['dollar_cost'] = \
        df['cpu_core_seconds'] * hourly_vcpu_cost/3600 \
        + df['memory_gb_seconds'] * hourly_gb_mem_cost/3600
    return df


def with_normalization(df, main_column, group_columns):
    grouped = df.groupby(['algorithm', 'dataset_size'])

    # Calculate the minimum <main_column> for each group
    df[f'min_{main_column}'] = grouped[main_column].transform('min')

    # Create the normalized_<main_column> column by dividing <main_column> by min_<main_column>
    df[f'normalized_{main_column}'] = df[main_column] / df[f'min_{main_column}']

    # Drop the 'min_...' column since it's not needed anymore
    df = df.drop(columns=f'min_{main_column}')
    return df

regular_dataproc_df = load_regular_dataproc_data(REGULAR_DATA_FILE)
serverless_dataproc_df, executor_counts = load_serverless_dataproc_data(SERVERLESS_DATA_DIR)

config_keys = ['scaleout', 'total_cluster_cpu_cores', 'total_cluster_memory_gb']
job_keys = ['algorithm', 'dataset_size']

comparison_keys = [  # attributes that exist for serverless & regular dataproc
    'algorithm',
    'dataset_size',
    'mode',
    'runtime',
    'cpu_core_seconds',
    'memory_gb_seconds',
    'dollar_cost',
    'mean_scaleout',
    'mean_total_cluster_cpu_cores',
    'mean_total_cluster_memory_gb',
    'config_name',
]
algorithms = set(regular_dataproc_df['algorithm'])
cluster_configs = regular_dataproc_df[config_keys].drop_duplicates()

def merge_dataproc_jobs(df_serverless, df_regular):

    attach_column(df_serverless, 'mode', value='serverless')
    attach_column(df_regular, 'mode', value='regular')
    attach_column(df_regular, 'mean_scaleout', values=df_regular['scaleout'])
    attach_column(df_regular, 'mean_total_cluster_cpu_cores',
                  values=df_regular['total_cluster_cpu_cores'])
    attach_column(df_regular, 'mean_total_cluster_memory_gb',
                  values=df_regular['total_cluster_memory_gb'])

    df_regular = with_dollar_cost(df_regular, HOURLY_VCPU_COST, HOURLY_GB_MEM_COST)
    df_serverless = with_dollar_cost(df_serverless, HOURLY_VCPU_COST, HOURLY_GB_MEM_COST)

    df3 = pd.concat((df_regular[comparison_keys], df_serverless[comparison_keys]), axis=0)
    df3.reset_index()
    return df3

df = merge_dataproc_jobs(serverless_dataproc_df, regular_dataproc_df)
df = with_normalization(df, 'dollar_cost', group_columns=['algorithm', 'dataset_size'])
df = with_normalization(df, 'runtime', group_columns=['algorithm', 'dataset_size'])

def explore_dataset(df):
    print(df[['runtime', 'cpu_core_seconds', 'memory_gb_seconds']].describe())

def compare_regular_vs_serverless():

    so8 = df['mean_scaleout'] == 8
    so16 = df['mean_scaleout'] == 16
    cpu32 = df['mean_total_cluster_cpu_cores'] == 32
    cpu64 = df['mean_total_cluster_cpu_cores'] == 64

    comparable = df[so8 & cpu32 | so16 & cpu64]\
            [['algorithm', 'dataset_size', 'mode', 'mean_scaleout', 'runtime']]

    stats = []  # (regular, serverless)
    for job, group in comparable.groupby(job_keys+['mean_scaleout']):
        g = with_normalization(group, 'runtime', job_keys+['mean_scaleout'])
        g = group.sort_values(['mode'])
        stats.append([
            get_row(g, mode='regular')['normalized_runtime'].mean(),
            get_row(g, mode='serverless')['normalized_runtime'].mean(),
        ])

    runtime_comparison = pd.DataFrame(stats, columns=['regular', 'serverless'])
    print((runtime_comparison['serverless'] / runtime_comparison['regular']).describe())

def compare_cost_of_all_configs():

    li = []
    for name, group in df.groupby(['config_name']):
        li.append((name[0], group['normalized_dollar_cost'].mean(), group['normalized_runtime'].mean()))
    results = pd.DataFrame(li, columns=('config_name', 'cost (normalized)', 'runtime (normalized)'))
    print(results)
    print("\nMean normalized cost:", results['cost (normalized)'].mean())
    print("Mean normalized runtime:", results['runtime (normalized)'].mean())

def plot_timeseries():

    label_size = 20
    legend_size = 18
    title_size = 18
    tick_size = 16

    def overlay(x1, y1, x2, y2, title, ax, i):
        ax.plot(x1, y1, linestyle=(0, (1,1)), alpha=1, linewidth=3, label='Smaller dataset (size$ := s_i$ )')
        ax.plot(x2, y2, linestyle=(1, (1,1)), alpha=1, linewidth=3, label='Larger dataset (size$ = 2 · s_i$)')
        ax.set_ylim(0,40)

        ax.set_yticks(np.arange(0,33, 8))
        if i % 2:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.grid(True, axis='y', color="#DDD", linewidth=0.25)
        ax.set_title(title, y=0.84,  fontweight='bold', color='dimgray', fontsize=title_size)

    fig, axs = plt.subplots(5, 2, figsize=(10, 14))  # 12 inches wide, 6 inches tall
    axs = axs.flat
    for i, algorithm in enumerate(sorted(algorithms)):
        df = executor_counts[f"{algorithm}-small-2-32"]
        ts = list(map(lambda x: round(x, 2), df['timestamp'].astype(float).tolist()))
        ec = df['executor_count'].astype(int).tolist()
        x1 = sum(zip(ts, ts[1:]), ()) + (ts[-1],)
        y1 = sum(zip(ec ,ec[:-1]), ()) + (0,)

        df = executor_counts[f"{algorithm}-large-2-32"]
        ts = list(map(lambda x: round(x, 2), df['timestamp'].astype(float).tolist()))
        ec = df['executor_count'].astype(int).tolist()
        x2 = sum(zip(ts, ts[1:]), ()) + (ts[-1],)
        y2 = sum(zip(ec ,ec[:-1]), ()) + (0,)

        overlay(x1, y1, x2, y2, f"{algorithm_names[algorithm]}", axs[i], i)

    axs[9].axis('off')

    fig.supxlabel('Time [seconds]', y=0.01, fontsize=label_size)
    fig.supylabel('Number of Executors', x= 0.015, fontsize=label_size)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(.76, 0.15), fontsize=legend_size)
    plt.tight_layout(h_pad=.4, w_pad=.4)  # can prevent overlap
    plt.savefig("plots/timeseries_5x2.svg", bbox_inches='tight')
    plt.savefig("plots/timeseries_5x2.pdf", bbox_inches='tight')
    plt.clf()


def show_efficiency(df):
    df = with_normalization(df, 'executor_seconds', group_columns=job_keys)

    # Uncomment to see per job details
    # print(get_rows(df, min_executors=2)[job_keys + ['normalized_executor_seconds']])
    # print(get_rows(df, min_executors=8)[job_keys + ['normalized_executor_seconds']])
    # print(get_rows(df, min_executors=16)[job_keys + ['normalized_executor_seconds']])

    print("Mean executor seconds [normalized] for 2-32 / 8 / 16 executors:")
    print(get_rows(df, min_executors=2)['normalized_executor_seconds'].mean())
    print(get_rows(df, min_executors=8)['normalized_executor_seconds'].mean())
    print(get_rows(df, min_executors=16)['normalized_executor_seconds'].mean())


def plot_cost_performance_tradeoff(df):
    title_size = 18
    label_size = 16
    legend_size = 14
    tick_size = 14
    marker_size = 50

    df = with_normalization(df, 'executor_seconds', group_columns=job_keys)
    df = with_normalization(df, 'runtime', group_columns=job_keys)
    df2 = get_rows(df, min_executors=2)
    df8 = get_rows(df, min_executors=8)
    df16 = get_rows(df, min_executors=16)
    plt.scatter(df2['normalized_executor_seconds'], df2['normalized_runtime'], s=marker_size*0.9, marker='x', label='S1', zorder=3, color='black')
    plt.scatter(df8['normalized_executor_seconds'], df8['normalized_runtime'], s=marker_size*1.5, marker='*', label='S2', color='tab:pink')
    plt.scatter(df16['normalized_executor_seconds'], df16['normalized_runtime'], s=marker_size, marker='o', label='S3', zorder=2, color='salmon')
    plt.xlabel('Expended executor seconds [normalized]', fontsize=label_size);
    plt.ylabel('Runtime [normalized]', fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.title('Execution Cost/Performance Relations', fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.savefig("plots/tradeoff.svg", bbox_inches='tight')
    plt.savefig("plots/tradeoff.pdf", bbox_inches='tight')
    plt.clf()


def run_mem_cpu_cost_ratio_experiment(results_file_name):

    res = []

    global HOURLY_VCPU_COST
    global HOURLY_GB_MEM_COST

    HOURLY_VCPU_COST = 1

    for mem_cost in np.arange(0.01, 11.01, 0.01):

        HOURLY_GB_MEM_COST = round(mem_cost, 2)

        regular_dataproc_df = load_regular_dataproc_data(REGULAR_DATA_FILE)
        serverless_dataproc_df, executor_counts = load_serverless_dataproc_data(SERVERLESS_DATA_DIR)
        df = merge_dataproc_jobs(serverless_dataproc_df, regular_dataproc_df)
        df = with_normalization(df, 'dollar_cost', group_columns=['algorithm', 'dataset_size'])
        # df = with_normalization(df, 'runtime', group_columns=['algorithm', 'dataset_size'])
        for name, group in df.groupby(['config_name']):
            res.append((name[0], f'{mem_cost:.02f}', group['normalized_dollar_cost'].mean()))

    columns = ["config_name", "mem_cpu_cost_ratio", "normalized_dollar_cost"]
    pd.DataFrame(res, columns=columns).to_csv(results_file_name, index=False)


def present_mem_cpu_cost_ratio_experiment(results_file_name):
    df = pd.read_csv(results_file_name)

    fat, very_thin = 3.5, 1.5
    line_properties = {
        'S1': ('black',   fat, (0,(3,1))),
        'S2': ('#7f7f7f', fat, (0,(3,1))),
        'S3': ('#bcbd22', fat, (0,(3,1))),
        'R01': (None,  very_thin, 'solid'),
        'R02': (None,  very_thin, 'solid'),
        'R03': (None,  very_thin, 'solid'),
        'R04': (None,  very_thin, 'solid'),
        'R05': (None,  very_thin, 'solid'),
        'R06': (None,  very_thin, 'solid'),
        'R07': (None,  very_thin, 'solid'),
        'R08': (None,  fat, 'solid'),
        'R09': (None,  fat, 'solid'),
        'R10': (None,  very_thin, 'solid'),
    }

    for config_name, group in df.groupby('config_name'):
        color, linewidth, linestyle = line_properties.get(config_name) or (None, None, None)
        plt.plot(
            group['mem_cpu_cost_ratio'],
            group['normalized_dollar_cost'],
            label=config_name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    plt.legend()
    plt.plot([0.134004, 0.134004], [0, 10], '-', linewidth=.5, color='green')
    plt.text(0.126, 0.985, '^A', verticalalignment='top', horizontalalignment='left', color='green', fontsize=9.5)
    plt.plot([0.16666, 0.16666], [0, 10], '-', linewidth=.5, color='green')
    plt.text(0.16, 0.985, '^B', verticalalignment='top', horizontalalignment='left', color='green', fontsize=9.5)
    plt.ylim((0.9999, 3.5))
    plt.xlim((1e-2, 1.1e1))
    plt.xscale('log')
    plt.xlabel('hourly_cost(1 GB RAM) ÷ hourly_cost(1 vCPU core)', fontsize=13)
    plt.ylabel('Monetary cost of execution [normalized]', fontsize=13)
    plt.savefig('plots/prices_experiment.svg', bbox_inches='tight')
    plt.savefig('plots/prices_experiment.pdf', bbox_inches='tight')
    plt.clf()


explore_dataset(df)
# plot_timeseries()
# compare_regular_vs_serverless()
# show_efficiency(serverless_dataproc_df)
# compare_cost_of_all_configs()
# plot_cost_performance_tradeoff(serverless_dataproc_df)
# run_mem_cpu_cost_ratio_experiment('cost_experiment.csv')  # Runs for about 3 minutes on my laptop
# present_mem_cpu_cost_ratio_experiment('cost_experiment.csv')

