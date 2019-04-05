def begin_accuracy():
    return "\\begin{table}[H] \n\\begin{tabular}{c|c|c|c|c|c|c|} \cline{2-7} \n& Architecture & No filter & UM & HE & CLAHE & DNLM \\\\ \\hline"

def end_accuracy(metric):
    return "\end{tabular} \n\caption{metric} \n\end{table}".replace("metric", metric)

def init_first_row_accuracy(mag):
    return "\multicolumn{1}{|c|}{\multirow{3}{*}" +  "{mag}}".replace("mag",mag)

def init_others_row_accuracy():
    return "\multicolumn{1}{|c|}{}"

def end_others_row_accuracy():
    return "\\\\ \cline{2-7}"

def end_final_row_accuracy():
    return "\\\\ \hline"

example_row = " & Densenet & 0 & 0 & 0 & 0 & 0"


#print(begin_accuracy())
#print(init_first_row_accuracy("40") + example_row + end_others_row_accuracy())
#print(init_others_row_accuracy() + example_row + end_final_row_accuracy())
#print(end_accuracy("pla"))