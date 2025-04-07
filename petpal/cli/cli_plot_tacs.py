"""
CLI module to plot TACs from a TACs folder created by petpal function write-tacs.
"""
import argparse
from ..visualizations.tac_plots import TacFigure, RegionalTacFigure
from ..utils.time_activity_curve import TimeActivityCurve


def main():
    """
    CLI for tac plotting
    """

    parser = argparse.ArgumentParser(prog='petpal-plot-tacs',
                                     description='Command line interface for plotting TACs.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tac-files',required=False,nargs='+')
    parser.add_argument('--tac-dir',required=False)
    parser.add_argument('--participant',required=False)
    parser.add_argument('--regions',required=False,nargs='+')
    parser.add_argument('--plot-type',
                        required=False,
                        default='both',
                        choices=['linear','log','both'])
    parser.add_argument('--yaxis-units',
                        required=False,
                        default='Bq/mL',
                        choices=['Bq/mL','kBq/mL','cps'])
    parser.add_argument('--xaxis-units',
                        required=False,
                        default='minutes',
                        choices=['minutes','seconds','hours'])
    parser.add_argument('--out-fig-path',required=True)
    args = parser.parse_args()


    if args.tac_dir is None:
        fig = TacFigure(plot_type=args.plot_type,
                        xlabel=fr'$t$ [{args.xaxis_units}]',
                        ylabel=fr'TAC [$\mathrm{{{args.yaxis_units}}}$]')
    else:
        fig = RegionalTacFigure(tacs_dir=args.tac_dir,
                                plot_type=args.plot_type,
                                xlabel=fr'$t$ [{args.xaxis_units}]',
                                ylabel=fr'TAC [$\mathrm{{{args.yaxis_units}}}$]')

    for tac_file in args.tac_files:
        tac = TimeActivityCurve.from_tsv(filename=tac_file)
        fig.add_errorbar(*tac.tac_werr)

    if args.tac_dir is not None:
        fig.plot_tacs_in_regions_list(regions=args.regions)

    if args.participant is not None:
        fig.fig.suptitle(t=args.participant)

    fig.write_fig(out_fig_path=args.out_fig_path)


if __name__ == "__main__":
    main()
