import pickle

from util import get_corpus_text
from build_network import build_network
from semcable.vsm_experiment import VSMResult, generate_barh_plots

tfox = get_corpus_text('Three Months for Semantic Network Experiments', 'FOXNEWSW')
tmsnbc = get_corpus_text('Three Months for Semantic Network Experiments', 'MSNBCW')

efox = build_network(tfox[0], alpha=0.75, verbose=True)

rf = VSMResult(efox[0], efox[1])
rf.make_graph()

emsnbc = build_network(tmsnbc[0], alpha=0.75, verbose=True)

rm = VSMResult(emsnbc[0], emsnbc[1])
rm.make_graph()

pickle.dump(rf, open('fox_result', 'wb'))
pickle.dump(rm, open('msnbc_result', 'wb'))

generate_barh_plots(
    '/home/mturner8/vsm-1-30/plots',
    ['swamp', 'epa', 'immigration', 'climate', 'liberal',
     'conservative', 'christian', 'muslim'], rm, rf
)
