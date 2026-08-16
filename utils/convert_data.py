from textwrap import dedent
from utils.file_tool import json_read_to_dict

def convert_user_data(data_dict):
    role_dict = data_dict['role']
    base_info = f"""
    # base_info
    ## demographic_information
    {role_dict['demographic_information']}
    ## lifestyle
    {role_dict['lifestyle']}
    ## medical_history
    {role_dict['medical_history']}
"""
    data_str = '# datas\n'

    for day_data in data_dict['data_s']:
        data_str += '## {}\n'.format(day_data['datetime'])
        data_str += '### sport data (Refers to the full-day exercise record data.)\n'
        for sport_dict in day_data['sport_s']:
            data_str += 'Start at {} and end at {}, do {} minutes of {} exercise, consume {} kilocalories.\n'.format(
               sport_dict['start_time'],
                    sport_dict['end_time'],
                    sport_dict['sport_time'],
                    sport_dict['sport_type'],
                    sport_dict['calorie'])
        data_str += '### health data (Refers to the full-day health monitoring data.)\n'
        for health_dict in day_data['health_s']:
            for af in health_dict['atrial_fibrillation']:
                data_str += '{} Detected {} time atrial fibrillation.\n'.format(af['record_time'], af['value'])
            for ph in health_dict['premature_heartbeat']:
                data_str += '{} Detected {} time premature heartbeat.\n'.format(ph['record_time'], ph['value'])
            for sa in health_dict['sleep_apnea']:
                data_str += '{} Detected {} time sleep apnea.\n'.format(sa['record_time'], sa['value'])
            data_str += ('Sleep onset time is {}, total sleep duration is {} minutes, and sleep score is {}.\n'
                           .format(health_dict['sleep_stage']['start_time'],
                                   health_dict['sleep_stage']['sleep_time'],
                                   health_dict['sleep_stage']['sleep_scope']))
        data_str += '### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\n'
        for activity_dict in day_data['activity_s']:
            all_day_step_count = activity_dict['all_day_step_count']
            all_day_sitting_num = activity_dict['all_day_sitting_num']
            all_day_calorie = activity_dict['all_day_calorie']
            data_str += ('the entire day saw a total of {} steps taken, {} sedentary periods, and a daily calorie consumption of {} kilocalories.\n'
                         .format(all_day_step_count, all_day_sitting_num, all_day_calorie))

    data = f"""
{data_str}
"""
    base_info = dedent(base_info)
    data = dedent(data)
    return base_info, data


if __name__ == '__main__':
    data_dir = '../data/roles_data_en_10000/role_data_1.json'
    data_dict = json_read_to_dict(data_dir)
    base_info, data = convert_user_data(data_dict)
    print(base_info)
    print(data)
