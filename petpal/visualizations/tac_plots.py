"""
Simple module to plot TACs from a TACs folder created by petpal function write-tacs.
"""
import argparse
from ..visualizations.tac_plots import tacs_to_df, tac_plots


def main():
    """
    CLI for tac plotting
    """

    parser = argparse.ArgumentParser(prog='petpal-plot-tacs',
                                    description='Command line interface for plotting TACs.',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tacs-folder',required=True)
    parser.add_argument('--participant',required=True)
    parser.add_argument('--regions',required=True,nargs='+')
    parser.add_argument('--tsv-save',required=True)
    parser.add_argument('--png-save',required=True)
    args = parser.parse_args()

    tacs_df = tacs_to_df(tacs_dir=args.tacs_folder,
                        participant=args.participant)
    tacs_df.to_csv(args.tsv_save)
    p = tac_plots(tacs_data=tacs_df,regions_to_plot=args.regions)
    p.get_figure().savefig(args.png_save)


if __name__ == "__main__":
    main()
