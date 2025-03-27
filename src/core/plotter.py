"""Defines plotting functions used for analyzing data.

See ``examples/analysis`` for examples of how to use.
"""

import glob
import json
import os
import warnings
import math

import matplotlib.colorbar as cbar
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from src.core import Query
from src.utils import Config, Object


def heatmaps(
    *facetdata,
    data=None,
    ax=None,
    return_metadata=False,  # add an explicit parameter for metadata output
    **kwargs,
):
    """Create heatmap for a single axis using the _HeatmapPlotter class.

    To create a single heatmap, call the class directly.
    Use a Seaborn ``FacetGrid`` to create multiple heatmaps in one figure, using the ``FacetGrid.map()`` method.
    Note that data cannot be aggregated across n_sims
    (e.g., each call of ``heatmaps()`` must receive only one threshold per fiber).

    :param facetdata: Receives data from ``FacetGrid`` if using to plot an array.
    :param data: DataFrame to plot, used if manually passing data.
    :param ax: Axis to plot on.
    :param return_metadata: Whether to return metadata along with the axis.
    :param kwargs: Arguments to be passed to the ``_HeatmapPlotter`` class constructor.
    :return: Plotting axis, or a tuple (axis, metadata) if ``return_metadata`` is True.
    """
    if data is None:
        data = pd.concat(facetdata, axis=1)
        
    # Pass the metadata flag explicitly to _HeatmapPlotter
    plotter = _HeatmapPlotter(data, return_metadata=return_metadata, **kwargs)
    
    if ax is None:
        ax = plt.gca()
    
    result = plotter.plot(ax)
    
    # If metadata was requested, the result is a tuple (ax, meta_data)
    if return_metadata:
        return result  # expected to be (ax, meta_data)
    else:
        return result  # just the axis


class _HeatmapPlotter:
    """Class used to contruct heatmap plots.

    This class should not be called directly by the user. Rather, the
    user should call the ``heatmaps()`` function, which will pass any
    keyword arguments to this class's constructor.
    """

    def __init__(
        self,
        data,
        mode: str = 'fibers',
        sample_object=None,
        sim_object=None,
        missing_color='red',
        suprathresh_color='blue',
        subthresh_color='green',
        cutoff_thresh=None,
        cmap=None,
        colorbar=True,
        min_max_ticks=False,
        cuff_orientation=False,
        plot_outers=False,
        cbar_kws=None,
        scatter_kws=None,
        line_kws=None,
        min_thresh=None,
        max_thresh=None,
        color=None,
        return_metadata=False,
    ):
        """Initialize heatmap plotter.

        :param data: DataFrame containing data to plot.
        :param mode: Plotting mode. There are multiple options:

            * ``'fibers'``: Plot a point for each fiber, using a heatmap of thresholds for color.
            * ``'fibers_on_off'``: Plot a point for each fiber. If the fiber threshold is above cutoff_thresh,
              suprathresh_color is used. Otherwise, subthresh_color is used.
            * ``'inners'``: Plot each inner as filled in, using a heatmap of thresholds for color.
              The mean threshold for that inner is used,
              thus if only one fiber is present per inner, that threshold is used.
            * ``'inners_on_off'``: Plot each inner as filled in. If the mean inner threshold is above cutoff_thresh,
              suprathresh_color is used. Otherwise, subthresh_color is used.

        :param sample_object: Sample object to use for plotting. Automatically loaded if not provided.
        :param sim_object: Simulation object to use for plotting. Automatically loaded if not provided.
        :param missing_color: Color to use for missing data.
        :param suprathresh_color: Color to use for suprathresh data.
        :param subthresh_color: Color to use for subthresh data.
        :param cutoff_thresh: Threshold to use for plotting on_off modes.
        :param cmap: Color map to override default.
        :param colorbar: Whether to add a colorbar.
        :param min_max_ticks: Whether to add only the minimum and maximum ticks to the colorbar.
        :param cuff_orientation: Whether to plot a point for the cuff orientation.
        :param plot_outers: Whether to plot the fascicle outers.
        :param cbar_kws: Keyword arguments to pass to matplotlib.colorbar.Colorbar.
        :param scatter_kws: Keyword arguments to pass to matplotlib.pyplot.scatter.
        :param line_kws: Keyword arguments to pass to matplotlib.pyplot.plot.
        :param min_thresh: Minimum threshold to use for plotting. Use this to override the default minimum.
        :param max_thresh: Maximum threshold to use for plotting. Use this to override the default maximum.
        :param color: Color passed in by seaborn when using FacetGrid. Not used.
        """
        # add variables to self from input args
        self.min_thresh = self.max_thresh = None
        self.mappable = None
        self.fiber_colors = self.inner_colors = None
        self.sample_index = self.sim_index = self.model_index = self.n_sim_index = None
        self.plot_outers = plot_outers
        self.cmap = cmap
        self.min_max_ticks = min_max_ticks
        self.colorbar = colorbar
        self.cutoff_thresh = cutoff_thresh
        self.missing_color = missing_color
        self.suprathresh_color = suprathresh_color
        self.subthresh_color = subthresh_color
        self.mode = mode
        self.sample = sample_object
        self.sim = sim_object
        self.color = color
        self.cbar_kws = cbar_kws if cbar_kws is not None else {}
        self.scatter_kws = scatter_kws if scatter_kws is not None else {}
        self.scatter_kws.setdefault('s', 100)
        self.line_kws = line_kws if line_kws is not None else {}
        self.max_thresh = max(data.threshold) if max_thresh is None else max_thresh
        self.min_thresh = min(data.threshold) if min_thresh is None else min_thresh
        self.cuff_orientation = cuff_orientation
        self.return_metadata = return_metadata

        # run setup in preparation for plotting
        self.validate(data)
        self.get_objects()
        self.create_cmap()
        self.determine_colors(data)

    def plot(self, ax):
        """Make heatmap plot.

        :param ax: Axis to plot on.
        :return: Plotting axis.
        """
        self.set_ax(ax)

        if self.colorbar and self.mode != 'on_off':
            self.add_colorbar(ax)

        if self.cuff_orientation and self.return_metadata is True:
            meta_data = self.plot_cuff_orientation(ax)
        elif self.cuff_orientation:
            self.plot_cuff_orientation(ax)
        else:
            pass  # Add a valid statement or logic here if needed
            

        self.plot_inners_fibers(ax)

        if self.return_metadata is True:
            return ax, meta_data
        else:
            return ax
        

    def plot_inners_fibers(self, ax):
        """Plot inners and fibers using the colors determined in determine_colors().

        :param ax: axis to plot on
        """
        self.sample.slides[0].plot(
            final=False,
            fix_aspect_ratio=True,
            fascicle_colors=self.inner_colors,
            ax=ax,
            outers_flag=self.plot_outers,
            inner_format='k-',
            scalebar=True,
            line_kws=self.line_kws,
        )
        if np.any([bool(x) for x in self.fiber_colors]):
            self.scatter_kws['c'] = self.fiber_colors
            self.sim.fibersets[0].plot(ax=ax, scatter_kws=self.scatter_kws)

    def create_cmap(self):
        """Create color map and mappable for assigning colorbar and ticks."""
        if self.cmap is None:
            cmap = plt.cm.get_cmap('viridis')
            cmap.set_bad(color='w')
            cmap = cmap.reversed()
            self.cmap = cmap
        mappable = plt.cm.ScalarMappable(
            cmap=self.cmap,
            norm=mplcolors.Normalize(vmin=self.min_thresh, vmax=self.max_thresh),
        )
        self.mappable = mappable

    @staticmethod
    def set_ax(ax):
        """Remove axis elements.

        :param ax: axis to plot on
        """
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    def determine_colors(self, threshdf):
        """Determine colors for inners and fibers based on user selected mode.

        :param threshdf: DataFrame of thresholds.
        """

        def _mapthresh(thresh):
            return tuple(self.cmap((thresh - self.min_thresh) / (self.max_thresh - self.min_thresh)))

        inner_color_list = []
        fiber_color_list = []
        for inner in pd.unique(threshdf.inner):
            # get inner threshold and add the appropriate color to the list
            innerthresh = np.mean(threshdf.query(f'inner=={inner}').threshold)
            if innerthresh is np.nan:
                inner_color_list.append(self.missing_color)
                warnings.warn(
                    'Missing at least one fiber threshold, color will appear as missing color (defaults to red).',
                    stacklevel=2,
                )
            elif self.mode == 'inners':
                inner_color_list.append(_mapthresh(innerthresh))
            elif self.mode == 'inners_on_off':
                inner_color_list.append(
                    self.suprathresh_color if innerthresh > self.cutoff_thresh else self.subthresh_color
                )
            else:
                inner_color_list.append(None)
        for fiber_index in pd.unique(threshdf['index']):
            # get fiber threshold and add the appropriate color to the list
            fiberthresh = np.mean(threshdf.query(f'index=={fiber_index}').threshold)
            if fiberthresh is np.nan:
                warnings.warn(
                    'Missing fiber threshold, color will appear as missing color (defaults to red).', stacklevel=2
                )
                if self.mode in ['fibers', 'fibers_on_off']:
                    fiber_color_list.append(self.missing_color)
                else:
                    fiber_color_list.append(None)
            elif self.mode == 'fibers':
                fiber_color_list.append(_mapthresh(fiberthresh))
            elif self.mode == 'fibers_on_off':
                fiber_color_list.append(
                    self.suprathresh_color if fiberthresh > self.cutoff_thresh else self.subthresh_color
                )
            else:
                fiber_color_list.append(None)
        # set colors for inners and fibers
        self.inner_colors, self.fiber_colors = inner_color_list, fiber_color_list

    def add_colorbar(self, ax):
        """Add colorbar to heatmap plot.

        :param ax: axis to plot on
        """
        # set default ticks if not provided
        if 'ticks' not in self.cbar_kws:
            self.cbar_kws['ticks'] = (
                tick.AutoLocator() if not self.min_max_ticks else [self.min_thresh, self.max_thresh]
            )
        # generate colorbar
        cb_label = r'mA'
        cb: cbar.Colorbar = plt.colorbar(mappable=self.mappable, ax=ax, **self.cbar_kws)
        cb.ax.set_title(cb_label)

    def get_objects(self):
        """Get sample and sim objects for plotting."""
        if self.sample is None:
            self.sample = Query.get_object(Object.SAMPLE, [self.sample_index])
        if self.sim is None:
            self.sim = Query.get_object(Object.SIMULATION, [self.sample_index, self.model_index, self.sim_index])

    def validate(self, data):
        """Check that data is valid for plotting.

        :param data: DataFrame of thresholds.
        """
        assert self.mode in ['fibers', 'inners', 'fibers_on_off', 'inners_on_off']
        if self.mode in ['fibers_on_off', 'inners_on_off']:
            assert self.cutoff_thresh is not None, 'Must provide cutoff threshold for on/off mode.'
        # make sure only one sample, model, sim, and nsim for this plot
        for index in ['sample', 'model', 'sim', 'nsim']:
            assert (
                len(pd.unique(data[index])) == 1
            ), f'Only one {index} allowed for this plot. Append something like q.threshold_data.query(\'{index}==0\')'
            setattr(self, index + '_index', pd.unique(data[index])[0])

    def plot_cuff_orientation(self, ax):
        """Plot the orientation of the cuff.

        :param ax: axis to plot on
        """
        # calculate orientation point location (i.e., contact location)
        # get radius of sample
        try:
            r = self.sample.slides[0].nerve.mean_radius()
            print(f'Radius: {r}')
        except AttributeError:
            r = self.sample.slides[0].fascicles[0].outer.mean_radius()
        # get orientation angle from slide
        theta = self.sample.slides[0].orientation_angle if self.sample.slides[0].orientation_angle is not None else 0
        # load add_ang from model.json cofiguration file
        with open(Query.build_path(Config.MODEL, [self.sample_index, self.model_index])) as f:
            model_config = json.load(f)
        # add any cuff rotation
        theta += np.deg2rad(model_config.get('cuff')[0].get('rotate').get('add_ang'))
        print(f'Orientation: {np.rad2deg(theta)} degrees, {theta} radians')
        ax.scatter(r * 1.2 * np.cos(theta), r * 1.2 * np.sin(theta), 300, 'red', 'o')
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        print(f'x: {x}, y: {y}')
        if self.return_metadata:
            meta_data = {}
            meta_data["orientation_point_x"] = x
            meta_data["orientation_point_y"] = y
            meta_data["r_cuff"] = r
            meta_data["cuff_shift_x"]= model_config.get('cuff')[0].get('shift').get('x')            # Cuff center x-coordinate
            meta_data["cuff_shift_y"]= model_config.get('cuff')[0].get('shift').get('y')   # Cuff center y-coordinate
            return meta_data
        


def ap_loctime(  # noqa: C901
    query_object: Query,
    n_sim_filter: list[int] = None,
    plot: bool = False,
    plot_distribution: bool = False,
    n_sim_label_override: str = None,
    model_labels: list[str] = None,
    save: bool = False,
    subplots=False,
    nodes_only=False,
    amp=0,
):
    """Plot time and location of action potential initiation.

    :param query_object: Query object to use for plotting.
    :param n_sim_filter: List of n_sim values to plot.
    :param plot: Whether to plot the ap location node for each fiber.
    :param plot_distribution: Whether to plot action potential initiation node distribution.
    :param n_sim_label_override: Label to use for n_sim.
    :param model_labels: Labels to use for models.
    :param save: Whether to save the plot.
    :param subplots: Whether to plot in subplots.
    :param nodes_only: Whether to plot only nodes.
    :param amp: Amplitude of action potential.
    """
    # loop samples
    for sample_index, sample_results in [(s['index'], s) for s in query_object._result.get('samples')]:
        print(f'sample: {sample_index}')

        # loop models
        for model_index, model_results in [(m['index'], m) for m in sample_results.get('models')]:
            print(f'\tmodel: {model_index}')

            # loop sims
            for sim_index in model_results.get('sims', []):
                print(f'\t\tsim: {sim_index}')

                sim_object = query_object.get_object(Object.SIMULATION, [sample_index, model_index, sim_index])

                if subplots is True:
                    fig, axs = plt.subplots(ncols=len(sim_object.master_product_indices), nrows=2, sharey="row")

                # loop nsims
                for n_sim_index, (potentials_product_index, _waveform_index) in enumerate(
                    sim_object.master_product_indices
                ):
                    print(f'\t\t\tnsim: {n_sim_index}')

                    _, *_, fiberset_index = sim_object.potentials_product[potentials_product_index]

                    # skip if not in existing n_sim filter
                    if n_sim_filter is not None and n_sim_index not in n_sim_filter:
                        print('\t\t\t\t(skip)')
                        continue

                    # directory of data for this (sample, model, sim)
                    sim_dir = query_object.build_path(
                        Object.SIMULATION, [sample_index, model_index, sim_index], just_directory=True
                    )

                    # directory for specific n_sim
                    n_sim_dir = os.path.join(sim_dir, 'n_sims', str(n_sim_index))

                    # directory of fiberset (i.e., points and potentials) associated with this n_sim
                    fiberset_dir = os.path.join(sim_dir, 'fibersets', str(fiberset_index))

                    # the simulation outputs for this n_sim
                    outputs_path = os.path.join(n_sim_dir, 'data', 'outputs')

                    # path of the first inner, first fiber vm(t) data
                    inner = 0
                    if plot_distribution:
                        fiber_indices = len(glob.glob(fiberset_dir + '/*.dat'))
                    else:
                        fiber_indices = [0]
                    ap_nodes = []
                    for fiber_index in fiber_indices:
                        vm_t_path = os.path.join(
                            outputs_path, f'ap_loctime_inner{inner}_fiber{fiber_index}_amp{amp}.dat'
                        )

                        # load vm(t) data (see path above)
                        # each row is a snapshot of the voltages at each node [mV]
                        # the first column is the time [ms]
                        # first row is holds column labels, so this is skipped (time, node0, node1, ...)
                        aploc_data = np.loadtxt(vm_t_path, skiprows=0)

                        aploc_data[np.where(aploc_data == 0)] = float('Inf')

                        time = min(aploc_data)

                        node = np.argmin(aploc_data)
                        ap_nodes.append(node)

                        # create message about AP time and location findings
                        message = f't: {time} ms, node: {node + 1} (of {len(aploc_data) + 2})'
                        if time != float('inf'):
                            print(f'\t\t\t\t\t\t{message}')
                        else:
                            print('No action potential occurred.')
                            continue

                        # plot the AP location with voltage trace
                        # create subplots
                        if plot or save:
                            print(f'\t\t\t\tinner: {inner} \n \t\t\t\t\tfiber: {fiber_index}')
                            if subplots is not True:
                                fig, axes = plt.subplots(1, 1)
                                axes = [axes]
                            else:
                                axes = [axs[0][n_sim_index], axs[1][n_sim_index]]
                            # load fiber coordinates
                            fiber = np.loadtxt(os.path.join(fiberset_dir, f'{fiber_index}.dat'), skiprows=1)
                            nodefiber = fiber[0::11, :]

                            # plot fiber coordinates in 2D
                            if nodes_only is not True:
                                axes[0].plot(fiber[:, 0], fiber[:, 2], 'b.', label='fiber')
                            else:
                                axes[0].plot(nodefiber[:, 0], nodefiber[:, 2], 'b.', label='fiber')

                            # plot AP location
                            axes[0].plot(fiber[11 * node, 0], fiber[11 * node, 2], 'r*', markersize=10)

                            # location display settings
                            n_sim_label = (
                                f'n_sim: {n_sim_index}' if (n_sim_label_override is None) else n_sim_label_override
                            )
                            model_label = '' if (model_labels is None) else f', {model_labels[model_index]}'
                            axes[0].set_xlabel('x location, µm')

                            axes[0].set_title(f'{n_sim_label}{model_label}')
                            if subplots is not True:
                                axes[0].legend(['fiber', f'AP ({message})'])
                            else:
                                axes[0].legend(['fiber', 'AP'])

                            plt.tight_layout()

                            # voltages display settings
                            if subplots is not True or n_sim_index == 0:
                                axes[0].set_ylabel('z location, µm')
                            plt.tight_layout()

                        # display
                        if save:
                            plt.savefig(
                                (
                                    f'out/analysis/ap_time_loc_{sample_index}_{model_index}_{sim_index}_{n_sim_index}_'
                                    f'inner{inner}_fiber{fiber_index}.png'
                                ),
                                dpi=300,
                            )

                        if plot:
                            plt.show()
                    if plot_distribution:
                        total_nodes = len(aploc_data) + 2
                        plt.hist(ap_nodes, range=(0, total_nodes), bins=total_nodes)
                        plt.title('Distribution of AP initiation node sites')
                        plt.xlabel('Node Index')
                        plt.ylabel('Node Count')
                        plt.show()
                        if save:
                            plt.savefig(
                                (f'out/analysis/ap_time_loc_distribution_{sample_index}_{model_index}_{sim_index}.png'),
                                dpi=300,
                            )


def activation_profile(
    *facetdata,
    data=None,
    ax=None,
    currents=None,
    cutoff_current=None,
    _return_plotter=False,  # Special flag to return the plotter object
    **kwargs,
):
    """Create activation profile visualization that shows which fibers are activated.

    This function visualizes which fibers are activated (red) vs not activated (grey) 
    based on whether their threshold is below or above a specified cutoff current.

    :param facetdata: Receives data from FacetGrid if using to plot an array.
    :param data: DataFrame to plot, used if manually passing data.
    :param ax: Axis to plot on.
    :param currents: Dictionary with 'contact1_uA' and 'contact2_uA' values.
    :param cutoff_current: Current threshold to determine activation (in μA).
    :param _return_plotter: Internal flag to return the plotter object for testing.
    :param kwargs: Arguments to be passed to the _ActivationProfilePlotter constructor.
    :return: Plotting axis, or the plotter object if _return_plotter is True.
    """
    if data is None:
        data = pd.concat(facetdata, axis=1)
    
    plotter = _ActivationProfilePlotter(data, currents=currents, cutoff_current=cutoff_current, **kwargs)
    
    if ax is None:
        ax = plt.gca()
    
    result_ax = plotter.plot(ax)
    
    # Return the plotter itself for testing or advanced usage if requested
    if _return_plotter:
        return plotter
    else:
        return result_ax


class _ActivationProfilePlotter:
    """Class used to construct activation profile plots that show which fibers are activated.

    This class should not be called directly by the user. Instead,
    use the activation_profile() function, which will pass any keyword
    arguments to this class's constructor.
    """

    def __init__(
        self,
        data,
        currents=None,
        cutoff_current=None,
        sample_object=None,
        sim_object=None,
        activated_color='red',
        non_activated_color='grey',
        title_format='Nerve Fiber Activation: Contact 1 = {contact1_uA}μA, Contact 2 = {contact2_uA}μA',
        contact_colors=None,
        show_contacts=True,
        colorbar=False,
        cuff_orientation=True,
        plot_outers=True,
        scatter_kws=None,
        line_kws=None,
        cuff_array=None,
        color=None,
    ):
        """Initialize activation profile plotter.

        :param data: DataFrame containing threshold data.
        :param currents: Dictionary with 'contact1_uA' and 'contact2_uA' values.
        :param cutoff_current: Current threshold to determine activation (in μA).
        :param sample_object: Sample object to use for plotting. Automatically loaded if not provided.
        :param sim_object: Simulation object to use for plotting. Automatically loaded if not provided.
        :param activated_color: Color for activated fibers.
        :param non_activated_color: Color for non-activated fibers.
        :param title_format: Format string for the plot title.
        :param contact_colors: List of colors for cuff contacts.
        :param show_contacts: Whether to show the cuff contacts.
        :param colorbar: Whether to add a colorbar (not typically used).
        :param cuff_orientation: Whether to plot a point for the cuff orientation.
        :param plot_outers: Whether to plot the fascicle outers.
        :param scatter_kws: Keyword arguments to pass to matplotlib.pyplot.scatter.
        :param line_kws: Keyword arguments to pass to matplotlib.pyplot.plot.
        :param cuff_array: Array specifying the current distribution across contacts.
        :param color: Color passed in by seaborn when using FacetGrid. Not used.
        """
        # Set default values for currents and cutoff if not provided
        self.currents = currents or {'contact1_uA': 400, 'contact2_uA': 0}
        self.cutoff_current = cutoff_current or self.currents.get('contact1_uA', 0) + self.currents.get('contact2_uA', 0)
        
        # Store configuration parameters
        self.sample_index = self.sim_index = self.model_index = self.n_sim_index = None
        self.plot_outers = plot_outers
        self.activated_color = activated_color
        self.non_activated_color = non_activated_color
        self.title_format = title_format
        self.contact_colors = contact_colors or ["red", "blue", "green", "orange", "purple", "teal"]
        self.show_contacts = show_contacts
        self.colorbar = colorbar
        self.cuff_orientation = cuff_orientation
        self.sample = sample_object
        self.sim = sim_object
        self.color = color
        
        # Set default scatter and line kwargs if not provided
        self.scatter_kws = scatter_kws if scatter_kws is not None else {}
        self.scatter_kws.setdefault('s', 100)
        self.line_kws = line_kws if line_kws is not None else {}
        
        # Cuff current array (what proportion of current goes to each contact)
        if self.currents.get('contact1_uA', 0) > 0 or self.currents.get('contact2_uA', 0) > 0:
            total = self.currents.get('contact1_uA', 0) + self.currents.get('contact2_uA', 0)
            if total > 0:
                # Default array: only first two contacts used with calculated proportions
                self.cuff_array = cuff_array or [
                    self.currents.get('contact1_uA', 0) / total if total > 0 else 0,
                    self.currents.get('contact2_uA', 0) / total if total > 0 else 0,
                    0, 0, 0, 0
                ]
            else:
                self.cuff_array = [0, 0, 0, 0, 0, 0]
        else:
            self.cuff_array = cuff_array or [0, 0, 0, 0, 0, 0]
            
        # Set default cuff parameters (will be calculated later if needed)
        self.cuff_deg = None
        self.cuff_r = None
        self.cuff_center_x = None
        self.cuff_center_y = None
        
        # Create lists to store fiber colors
        self.fiber_colors = []
        
        # Run setup
        self.validate(data)
        self.get_objects()
        self.determine_colors(data)

    def plot(self, ax):
        """Make activation profile plot.

        :param ax: Axis to plot on.
        :return: Plotting axis.
        """
        self.set_ax(ax)
        
        # Draw fascicles and fibers
        self.plot_inners_fibers(ax)
        
        # Draw cuff contacts if requested
        if self.show_contacts:
            self.plot_cuff_contacts(ax)
            
        # Add color legend
        self.add_legend(ax)
        
        # Add title with current values
        title = self.title_format.format(**self.currents)
        ax.set_title(title)
        
        # Add annotation showing total current
        ax.text(0.02, 0.02, f"Total Current: {self.cutoff_current}μA", transform=ax.transAxes)
        
        return ax

    def plot_inners_fibers(self, ax):
        """Plot inners and fibers using the colors determined in determine_colors().

        :param ax: axis to plot on
        """
        # Plot fascicle structure
        self.sample.slides[0].plot(
            final=False,
            fix_aspect_ratio=True,
            fascicle_colors=None, # No inner coloring for activation profile
            ax=ax,
            outers_flag=self.plot_outers,
            inner_format='k-',
            scalebar=True,
            line_kws=self.line_kws,
        )
        
        # Plot fibers with activation colors
        self.scatter_kws['c'] = self.fiber_colors
        self.sim.fibersets[0].plot(ax=ax, scatter_kws=self.scatter_kws)

    def plot_cuff_contacts(self, ax):
        """Plot the cuff contacts based on current distribution.

        :param ax: axis to plot on
        """
        # Get cuff parameters if not already calculated
        if self.cuff_deg is None:
            self.calculate_cuff_parameters()
        
        # Calculate angles for contacts
        theta = self.sample.slides[0].orientation_angle if self.sample.slides[0].orientation_angle is not None else 0
        
        # Draw the cuff circle
        cuff = mpatches.Arc(
            xy=(self.cuff_center_x, self.cuff_center_y),
            width=2 * self.cuff_r,
            height=2 * self.cuff_r,
            angle=0.0,
            theta1=0,
            theta2=360,
            color="gray",
            linewidth=1,
        )
        ax.add_artist(cuff)
        
        # Calculate orientation point coordinates
        try:
            # First try to get the nerve center
            nerve_x, nerve_y = 0, 0  # Default to origin
            if hasattr(self.sample.slides[0], 'nerve'):
                nerve_x, nerve_y = self.sample.slides[0].nerve.center
            
            # Use orientation angle for rotation
            rot_def = np.arctan2(nerve_y, nerve_x) + theta
        except Exception as e:
            # Fallback to a default orientation
            print(f"Warning: Could not determine nerve center: {e}")
            rot_def = theta
        
        const = 0.0
        ln_angs = [
            rot_def - 2.5 * self.cuff_deg - const,
            rot_def - 1.5 * self.cuff_deg - const,
            rot_def - 0.5 * self.cuff_deg - const,
            rot_def + 0.5 * self.cuff_deg - const,
            rot_def + 1.5 * self.cuff_deg - const,
            rot_def + 2.5 * self.cuff_deg - const,
        ]
        
        # Plot each contact with current
        for i, current in enumerate(self.cuff_array):
            if current > 0:
                active_contact = mpatches.Arc(
                    xy=(self.cuff_center_x, self.cuff_center_y),
                    width=2 * self.cuff_r + 220,
                    height=2 * self.cuff_r + 220,
                    angle=0.0,
                    theta1=math.degrees(ln_angs[i]) - self.cuff_span / 2,
                    theta2=math.degrees(ln_angs[i]) + self.cuff_span / 2,
                    color=self.contact_colors[i % len(self.contact_colors)],
                    linewidth=4,
                    alpha=current  # Set alpha based on current proportion
                )
                ax.add_artist(active_contact)
                
    def calculate_cuff_parameters(self):
        """Calculate ImThera cuff parameters based on sample radius."""
        try:
            # Try to get radius from nerve
            if hasattr(self.sample.slides[0], 'nerve'):
                r = self.sample.slides[0].nerve.mean_radius()
            # If no nerve, try fascicles
            elif hasattr(self.sample.slides[0], 'fascicles') and len(self.sample.slides[0].fascicles) > 0:
                r = self.sample.slides[0].fascicles[0].outer.mean_radius()
            else:
                # Fallback to default
                print("Warning: Could not determine cuff radius, using default.")
                r = 1800  # Default radius in μm
        except Exception as e:
            print(f"Warning: Error calculating cuff radius: {e}")
            r = 1800  # Default radius in μm
        
        # Convert to inches
        r_in_iti = r / 25400
        
        # Constants from ImThera cuff design
        r_cuff_in_pre_iti = 0.059  # inch (pre-expansion inner radius)
        ang_cuffseam_contactcenter_pre_itc = 53  # degrees
        ang_contactcenter_contactcenter_pre_itc = 51  # degrees
        contact_diameter_inch = 2 / 25.4  # Convert 2mm to inches

        # Calculate angular spacing between contacts
        ang_spacing_deg = ang_contactcenter_contactcenter_pre_itc * (r_cuff_in_pre_iti / r_in_iti)
        self.cuff_deg = math.radians(ang_spacing_deg)
        
        # Calculate contact span
        self.cuff_span = math.degrees(contact_diameter_inch / r_in_iti)
        
        # Set cuff radius and center
        self.cuff_r = r
        self.cuff_center_x = 0
        self.cuff_center_y = 0
                
    def add_legend(self, ax):
        """Add legend to activation profile plot.

        :param ax: axis to plot on
        """
        # Create legend patches for activated and non-activated
        activated_patch = mpatches.Patch(color=self.activated_color, label='Activated')
        non_activated_patch = mpatches.Patch(color=self.non_activated_color, label='Not Activated')
        
        # Add legend to plot
        ax.legend(handles=[activated_patch, non_activated_patch], loc='lower right')

    @staticmethod
    def set_ax(ax):
        """Remove axis elements for cleaner visualization.

        :param ax: axis to plot on
        """
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal')

    def determine_colors(self, threshdf):
        """Determine fiber colors based on activation status.

        :param threshdf: DataFrame of thresholds.
        """
        # Create a color list for each fiber
        self.fiber_colors = []
        
        # Track counts for activated and non-activated fibers
        activated_count = 0
        total_count = 0
        
        # Store activation info for detailed reporting
        self.activation_info = []
        
        # Convert cutoff current from μA to mA for proper comparison with thresholds
        cutoff_current_mA = self.cutoff_current / 1000.0  # Convert μA to mA
        
        print(f"Fiber Activation Analysis - Cutoff Current: {self.cutoff_current} μA ({cutoff_current_mA} mA)")
        print("-" * 80)
        
        # Try to get fiber positions from the sample object
        fiber_positions = {}
        try:
            # Get fiber coordinates from the sample's fiber set if available
            if hasattr(self.sample, 'fiber_set') and self.sample.fiber_set and self.sample.fiber_set.fibers:
                for idx, fiber in enumerate(self.sample.fiber_set.fibers):
                    if isinstance(fiber, dict) and 'fiber' in fiber:
                        # Take the first point's x,y coordinates
                        x, y, _ = fiber['fiber'][0]
                        fiber_positions[idx] = (x, y)
                    elif isinstance(fiber, list) and len(fiber) > 0:
                        # Take the first point's x,y coordinates
                        x, y, _ = fiber[0]
                        fiber_positions[idx] = (x, y)
        except Exception as e:
            print(f"Warning: Could not get fiber positions from sample: {e}")
        
        # For each fiber, check if it's activated based on threshold vs cutoff_current
        for fiber_index in pd.unique(threshdf['index']):
            fiber_data = threshdf.query(f'index=={fiber_index}')
            fiber_thresh = np.mean(fiber_data.threshold)
            
            # Get fiber coordinates if available from either DataFrame or fiber positions dictionary
            fiber_x = None
            fiber_y = None
            if 'pos_x' in fiber_data.columns and 'pos_y' in fiber_data.columns:
                fiber_x = fiber_data['pos_x'].iloc[0]
                fiber_y = fiber_data['pos_y'].iloc[0]
            elif fiber_index in fiber_positions:
                fiber_x, fiber_y = fiber_positions[fiber_index]
            
            # Store inner information
            inner = fiber_data['inner'].iloc[0] if 'inner' in fiber_data.columns else None
            
            # Determine activation status
            total_count += 1
            is_activated = False
            
            # If threshold is NaN, use non-activated color
            if np.isnan(fiber_thresh):
                warnings.warn(
                    'Missing fiber threshold, color will appear as missing color (defaults to grey).', 
                    stacklevel=2
                )
                self.fiber_colors.append(self.non_activated_color)
            else:
                # Proper comparison with converted units (mA)
                is_activated = fiber_thresh <= cutoff_current_mA
                self.fiber_colors.append(
                    self.activated_color if is_activated else self.non_activated_color
                )
                if is_activated:
                    activated_count += 1
            
            # Store activation info
            self.activation_info.append({
                'fiber_index': fiber_index,
                'inner': inner,
                'pos_x': fiber_x, 
                'pos_y': fiber_y,
                'threshold_mA': fiber_thresh,
                'is_activated': is_activated
            })
        
        # Print summary statistics
        self.activation_df = pd.DataFrame(self.activation_info)
        print(f"Activation Summary: {activated_count}/{total_count} fibers activated ({100.0*activated_count/total_count:.1f}%)")
        
        # Return activation counts for external use
        return activated_count, total_count

    def get_objects(self):
        """Get sample and sim objects for plotting."""
        if self.sample is None:
            self.sample = Query.get_object(Object.SAMPLE, [self.sample_index])
        if self.sim is None:
            self.sim = Query.get_object(Object.SIMULATION, [self.sample_index, self.model_index, self.sim_index])

    def validate(self, data):
        """Check that data is valid for plotting.

        :param data: DataFrame of thresholds.
        """
        # Make sure only one sample, model, sim, and nsim for this plot
        for index in ['sample', 'model', 'sim', 'nsim']:
            assert (
                len(pd.unique(data[index])) == 1
            ), f'Only one {index} allowed for this plot. Append something like q.threshold_data.query(\'{index}==0\')'
            setattr(self, index + '_index', pd.unique(data[index])[0])
