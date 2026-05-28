import numpy as np
import json

def print_double_tex_reslts_table(models_dict_left : dict, header_left : str,
                                  models_dict_right : dict, header_right : str,
                                  caption : str, label : str, metric_headers : str,
                                  header_len : str='{3}') -> None:
    metrics_left = list(list(models_dict_left.values())[0].keys())
    metrics_right = list(list(models_dict_right.values())[0].keys())
    models = list(models_dict_left.keys()) # both dicts must have the same models
    tex_table_string = f"""\\begin{{table*}}
    \\centering
    \\caption{caption}
    \\label{{table}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    %\\begin{{tabular}}{{|p{{25pt}}|p{{75pt}}|p{{115pt}}|}}
    \\resizebox{{\\textwidth}}{{!}}""" + """{""" + f"""\\begin{{tabular}}""" + """{{|l|"""

    for i in range(2*int(header_len[1])):
        tex_table_string += f"""l|"""
    tex_table_string += """}}"""
    
    tex_table_string += f"""
    \\hline
    \\multirow{{2}}{{*}}{{Model}} & \\multicolumn{header_len}{{|l|}}{header_left} & \\multicolumn{header_len}{{|l|}}{header_right} \\\\
    \\cline{{2-{str(int(header_len[1])*2+1)}}}
    & """ + metric_headers + """ & """ + metric_headers + """ \\\\
    \\hline"""
    for model in models:
        row_name = '{' + model.replace('_',' ') + '}'
        tex_table_string += f"""\\multirow{{2}}{{*}}{row_name} & ${models_dict_left[model][metrics_left[0]][0]:.3f}$"""
        for metric in metrics_left[1:]:
            tex_table_string += f""" & ${models_dict_left[model][metric][0]:.3f}$"""
        tex_table_string += f"""\n"""
        for metric in metrics_right:
            tex_table_string += f""" & ${models_dict_right[model][metric][0]:.3f}$"""
        tex_table_string += f""" \\\\ \n"""
        for metric in metrics_left:
            tex_table_string += f""" & $\\pm{models_dict_left[model][metric][1]:.3f}$"""
        tex_table_string += f"""\n"""
        for metric in metrics_right:
            tex_table_string += f""" & $\\pm{models_dict_right[model][metric][1]:.3f}$"""
        tex_table_string += f""" \\\\ 
        \\hline"""
    tex_table_string += f"""\\end{{tabular}}""" + """}""" + f"""
    \\label{label}
    \\end{{table*}}
    
    """
    
    print(tex_table_string)