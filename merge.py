import pandas as pd
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description='Process input CSV and output the results.')

# 添加命令行参数
parser.add_argument('input_csv', type=str, help='The input CSV file path.')
parser.add_argument('output_csv', type=str, help='The output CSV file path.')

# 解析命令行参数
args = parser.parse_args()

# 获取输入和输出文件路径
input_csv = args.input_csv
output_csv = args.output_csv
# 读取四个csv文件
ap_df = pd.read_csv('ASSESSMENT AND PLAN.csv')

hpi_df = pd.read_csv('HISTORY OF PRESENT ILLNESS.csv')
results_df = pd.read_csv('RESULTS.csv')
physical_exam_df = pd.read_csv('PHYSICAL EXAM.csv')

# 给每个DataFrame添加相应的标识符
hpi_df = hpi_df.add_prefix('| subjective |')
physical_exam_df = physical_exam_df.add_prefix('| objective_exam |')

results_df = results_df.add_prefix('| objective_results |')
ap_df = ap_df.add_prefix('| assessment_and_plan |')

# 按行方向进行合并
merged_df = pd.concat([hpi_df,physical_exam_df, results_df,ap_df ], axis=1)

# 
merged_df = merged_df.apply(lambda x: '\n'.join([f"{col_name[:-1]}{str(val)}" if pd.notna(val) else f"{col_name[:-1]}none." for col_name, val in x.iteritems()]), axis=1)

# 读取inputConversations.csv文件，并根据encounter_id列对merged_df和input_df进行内连接
input_df = pd.read_csv(input_csv)
new_df = pd.merge(input_df[['encounter_id']].rename(columns={'encounter_id': 'TestID'}), merged_df.to_frame(name='SystemOutput'), left_index=True, right_index=True, how='inner')

# 将结果保存到新的csv文件中
new_df.to_csv(output_csv, index=False)
