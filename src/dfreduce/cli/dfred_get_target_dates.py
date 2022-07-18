# Third-party
import numpy as np
import pandas as pd
from astropy.table import Table

# Project
from .. import utils
from ..database import DFIndividualFrames 


def get_target_dates(args, survey='UW'):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    targets = utils.parse_strings_or_file_to_list(args.targets, args.targets_fn)

    db = DFIndividualFrames(config['db_path'], config['data_path'])

    date_q = [f"target=='{t}'" for t in targets]
    q = "frame_type=='light' & (" + ' | '.join(date_q) + ')'

    results = db.query('target,date', where=q)

    targets = pd.unique(results.target).tolist()
    targets = [t for t in targets if survey.upper() in t.upper()]

    dates = [results.query(f"target == '{t}'").date for t in targets]
    dates = [','.join(np.unique(d)) for d in dates]

    radec_fetcher = utils.survey_radec_fetcher[survey.upper()]
    radec = [radec_fetcher(t) for t in targets]
    radec = [c.to_string('hmsdms') for c in radec]

    target_table = Table(data=[targets, radec, dates], 
                         names=['target', 'radec', 'dates'])

    return target_table


def main():
    parser = utils.default_script_parser('target')
    parser.add_argument('--survey', default='UW', type=str,
                        help='only targets from this survey')
    parser.add_argument('-o', '--out-fn', default=None, type=str, 
                        help='write target names to this csv file')

    args = parser.parse_args()
    targets = get_target_dates(args, args.survey)

    if args.out_fn is not None:
        targets.write(args.out_fn, overwrite=True)
    else:
        targets.pprint_all(align='<')
