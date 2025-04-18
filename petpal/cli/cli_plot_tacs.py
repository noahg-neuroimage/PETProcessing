"""
CLI module to plot TACs from a TACs folder created by petpal function write-tacs.
"""
import argparse

from ..visualizations.tac_plots import TacFigure, RegionalTacFigure
from ..utils.time_activity_curve import TimeActivityCurve


_PLOT_EXAMPLES_ = r"""
Examples:
  - Plot a single TAC:
    petpal-plot-tacs --tac-files my_tac.tsv --out-fig-path tac.png
  - Plot two or more TACs:
    petpal-plot-tacs --tac-files my_tac_1.tsv my_tac_2.tsv --out-fig-path tac.png
  - Plot all the TACs in a directory:
    petpal-plot-tacs --tac-dir sub-001/tacs/ --out-fig-path tac.png
  - Plot specific regional TACs in a directory based on region names:
    petpal-plot-tacs --tac-files my_tac.tsv --regions RightPutamen LeftPutamen --out-fig-path tac.png
  - Set x-axis and y-axis units:
    petpal-plot-tacs --tac-files my_tac.tsv --yaxis-units cps --xaxis-units hours --out-fig-path tac.png
  - Plot the linear-linear plot only:
    petpal-plot-tacs --tac-files my_tac.tsv --plot-type linear --out-fig-path tac.png
  - Set the figure title to the participant name:
    petpal-plot-tacs --tac-files my_tac.tsv --fig-title sub-001 --out-fig-path tac.png
"""


def main():
    """
    CLI for tac plotting
    """

    parser = argparse.ArgumentParser(prog='petpal-plot-tacs',
                                     description='Command line interface for plotting TACs.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=_PLOT_EXAMPLES_)
    parser.add_argument('--tac-files',
                        required=False,
                        nargs='+',
                        help='Path to one or more individual .tsv TAC files, separate paths with '
                             'spaces. At least one of: --tac-files, --tac-dir, is required.')
    parser.add_argument('--tac-dir',
                        required=False,
                        help='Path to a directory containing .tsv TAC files generated with PETPAL.'
                             ' At least one of: --tac-files, --tac-dir, is required.')
    parser.add_argument('--out-fig-path',
                        required=True,
                        help='Path to the file where the figure is saved.')
    parser.add_argument('--fig-title',
                        required=False,
                        help='Sets the figure title.')
    parser.add_argument('--regions',
                        required=False,
                        nargs='+',
                        help='If --tac-files is set, list the regions to be plotted. Separate '
                             'region names with spaces. Expecting TAC file names to follow the '
                             'convention: *seg-SegmentName* where SegmentName does not contain '
                             'special characters, especially - and _ which will conflict with '
                             'PETPAL code.')
    parser.add_argument('--plot-type',
                        required=False,
                        default='both',
                        choices=['linear','log','both'],
                        help='Set whether to plot the TACs as linear-linear, log-linear, or both.')
    parser.add_argument('--yaxis-units',
                        required=False,
                        default='Bq/mL',
                        choices=['Bq/mL','kBq/mL','cps','nCi/mL'],
                        help='Set activity concentration unit label for the y-axis. Does not scale'
                             ' units, this only assigns the axis label name.')
    parser.add_argument('--xaxis-units',
                        required=False,
                        default='minutes',
                        choices=['minutes','seconds','hours'],
                        help='Set time units for the x-axis. Does not scale units, this only '
                             'assigns the axis label name.')
    parser.add_argument('--plot-style',
                        required=False,
                        default='lines',
                        choices=['lines','markers','both'],
                        help='Set style of TAC plots.')
    parser.add_argument('--errorbars',
                        required=False,
                        action=argparse.BooleanOptionalAction,
                        help='Optional setting that adds errorbars to plots, if uncertainties are'
                             'available in the TAC files.')

    args = parser.parse_args()
    all_tac_plot_func = None
    regions_plot_func = None


    if args.tac_dir is None and args.tac_files is None:
        raise SystemExit('Both --tac-files and --tac-dir unset. Exiting.')

    if args.tac_dir is None:
        fig = TacFigure(plot_type=args.plot_type,
                        xlabel=fr'$t$ [{args.xaxis_units}]',
                        ylabel=fr'TAC [$\mathrm{{{args.yaxis_units}}}$]')
    else:
        fig = RegionalTacFigure(tacs_dir=args.tac_dir,
                                plot_type=args.plot_type,
                                xlabel=fr'$t$ [{args.xaxis_units}]',
                                ylabel=fr'TAC [$\mathrm{{{args.yaxis_units}}}$]')
        if args.errorbars is None:
            all_tac_plot_func = fig.plot_all_regional_tacs
            regions_plot_func = fig.plot_tacs_in_regions_list
        else:
            all_tac_plot_func = fig.plot_all_regional_tacs_with_errorbar
            regions_plot_func = fig.plot_tacs_in_regions_list_with_errorbar

    plot_style_opts = {'lines': {'marker':'None','ls':'-'},
                       'markers': {'marker':'.','ls':'None'},
                       'both': {'marker':'.','ls':'-'}}
    plot_style = plot_style_opts[args.plot_style]

    if args.tac_files is not None:
        if args.errorbars is None:
            for tac_file in args.tac_files:
                tac = TimeActivityCurve.from_tsv(filename=tac_file)
                fig.add_tac(*tac.tac, **plot_style)
        else:
            for tac_file in args.tac_files:
                tac = TimeActivityCurve.from_tsv(filename=tac_file)
                fig.add_errorbar(*tac.tac_werr, **plot_style)

    if args.tac_dir is not None:
        if args.regions is None:
            all_tac_plot_func(**plot_style)
        else:
            regions_plot_func(regions=args.regions, **plot_style)

    if args.fig_title is not None:
        fig.fig.suptitle(t=args.fig_title)

    fig.write_fig(out_fig_path=args.out_fig_path)


if __name__ == "__main__":
    main()
