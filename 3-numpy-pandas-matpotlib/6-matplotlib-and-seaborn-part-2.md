# 1 - Introduction
Bivariate Visualizations
  - two variables
  - how do the variables move in relation to each other

# 2 - Scatterplots and Correlation
Use to compare 2 quantitative variables
  - values for 1 are plotted on x
  - values for the other on on y

```
plt.scatter(data = df, x = 'num_var1', y = 'num_var2')
```

Seaborns `regplot` combines scatterplot with regression function fitting

```
sb.regplot(data = df, x = 'num_var1', y = 'num_var2')
```
  - regression only plots linear so if it's not a linear relationship it may not be right to include it
  - turn it off with `reg_fit = False`

# 3 - Overplotting, Transparency, and Jitter
overplotting - high amount of overlap in points (makes it difficult to see actual relationship)

Transparency can be added to a `scatter` by adding "alpha" parameter
  - image will be darker where more points overlap

```
plt.scatter(data = df, x = 'disc_var1', y = 'disc_var2', alpha = 1/5)
```

Jitter moves the position of each point slightly from its true value
  - allows you to see less overlap
  - can be used in conjunctio with transparency

```
sb.regplot(data = df, x = 'disc_var1', y = 'disc_var2', fit_reg = False,
           x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})
```

# 4 - Heat Maps
2D version of histogram that can be used as alternative to scatterplot
  - divided into grid of scales
  - number of data points in each grid cell == darker color

```
plt.figure(figsize = [12, 5])

# left plot: scatterplot of discrete data with jitter and transparency
plt.subplot(1, 2, 1)
sb.regplot(data = df, x = 'disc_var1', y = 'disc_var2', fit_reg = False,
           x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})

# right plot: heat map with bin edges between values
plt.subplot(1, 2, 2)
bins_x = np.arange(0.5, 10.5+1, 1)
bins_y = np.arange(-0.5, 10.5+1, 1)
plt.hist2d(data = df, x = 'disc_var1', y = 'disc_var2',
           bins = [bins_x, bins_y])
plt.colorbar();
```
  - because there are two vars, "bins" parm takes two bin edge specifications (as list)
  - choosing appropriate bin size is also important (same as with historgram)
  - `colorbar` adds a colorbar to side of plot showing mapping from counts to colors

Change the color palette with `cmap` param in `hist2D`
  - `cmap = 'viridis_r'`

Turn of bins with zero counts

```
bins_x = np.arange(0.5, 10.5+1, 1)
bins_y = np.arange(-0.5, 10.5+1, 1)
plt.hist2d(data = df, x = 'disc_var1', y = 'disc_var2',
           bins = [bins_x, bins_y], cmap = 'viridis_r', cmin = 0.5)
plt.colorbar()
```

You can add annotiations to the cells in the plot indicating count of points in each cell
  - `hist2d` has to add text elements one by one
  - look at [this example](https://classroom.udacity.com/nanodegrees/nd089/parts/8de94dee-7635-43b3-9d11-5e4583f22ce3/modules/dd3e4af8-d576-427e-baae-925fd16ff2ff/lessons/0491d74e-dcd8-4700-a971-a7f1b0a26ddb/concepts/8f54c142-468e-4e1a-abef-4c6425403a3b)

# 5 - Scatterplot Practice
Made some scatterplots and some heatmaps

# 6 - Violin Plots
Compare quantitative var vs qualitative var
  - where the curve is wider there are more data points
  - set color param to make each curve the same color if it's not meaningful
  - to remove the box plot you can set `inner = None`
  - seaborn automatically chooses orientation but if it's not what you like, pass in `orient`

```
sb.violinplot(data = df, x = 'cat_var', y = 'num_var')
```

# 7 - Box Plots
Another way of showing relationship between numeric and categorical ariable
  - leans more on summarization of data (compared to violin plot)
  - central line indicates median of the distribution
  - top/bottom represent third and first quartiles
  - whiskers indicate largest and smallest values
  - if you have lots of groups to compare, box plot is better than violin

```
base_color = sb.color_palette()[0]

sb.boxplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)
```

# 8 - Violin and Box Plot Practice
Made a violin plot

# 9 - Clustered Bar Charts
Use for qualitative var vs qualitative var

`sb.countplot(data = df, x = 'cat_var1', hue = 'cat_var2')`

Legend can be distracting so you can use axes method to set legend properties on axes

```
ax = sb.countplot(data = df, x = 'cat_var1', hue = 'cat_var2')
ax.legend(loc = 8, ncol = 3, framealpha = 1, title = 'cat_var2')
```

An alternative is to use a heatmap

```
# Summarize counts into matrix that will then be passed into heatmap to be plotted

ct_counts = df.groupby(['cat_var1', 'cat_var2']).size()
ct_counts = ct_counts.reset_index(name = 'count')
ct_counts = ct_counts.pivot(index = 'cat_var2', columns = 'cat_var1', values = 'count')
sb.heatmap(ct_counts)
```

# 10 - Categorical Plot Practice
Made a clustered bar chart.

# 11 - Faceting
Faceting is useful when you're handling plots of two or more variables.
  - data is divided into disjoint subsets (usually by different levels of categorical variable)
  - For each subset of data, the same plot type is rendered on other variables
  * helps compare distributions/relationships across levels of additional variables

```
g = sb.FacetGrid(data = df, col = 'cat_var')
g.map(plt.hist, "num_var")
```
- in `map` call just set plotting function and variable to be plotted as positional args. don't use keyword args or mapping wont work.

```
bin_edges = np.arange(-3, df['num_var'].max()+1/3, 1/3)
g = sb.FacetGrid(data = df, col = 'cat_var')
g.map(plt.hist, "num_var", bins = bin_edges)
```

If you have a lot of levels to plot, you may want to add more args to FacetGrid

Here is an example that creates 15 facet grids

```
group_means = df.groupby(['many_cat_var']).mean()
group_order = group_means.sort_values(['num_var'], ascending = False).index

g = sb.FacetGrid(data = df, col = 'many_cat_var', col_wrap = 5, size = 2,
                 col_order = group_order)
g.map(plt.hist, 'num_var', bins = np.arange(5, 15+1, 1))
g.set_titles('{col_name}')
```

# 12 - Adaptation of Univariate Plots
Adapted Bar Charts
  - can be adapted to be used as a bivariate plot (instead of indicating count by height it can indicate mean/statistic on 2nd var)

```
base_color = sb.color_palette()[0]
sb.barplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)
```
  - bar heights indicate mean value on numeric variable
  - error bars show uncertainty in mean based on variance and sample size

You can use `pointplot` to plot the averages as points rather than bars

```
sb.pointplot(data = df, x = 'cat_var', y = 'num_var', linestyles = "")
plt.ylabel('Avg. value of num_var')
```

  - by default pointplot connects values by a line, if you don't want that, pass in `linestyles = ""`

Adapted Histograms
  - can be adapted so bar heights indicate value other than count of points through use of the `weights` param
  - each data point is given weight of 1 by default

```
bin_edges = np.arange(0, df['num_var'].max()+1/3, 1/3)

# count number of points in each bin
bin_idxs = pd.cut(df['num_var'], bin_edges, right = False, include_lowest = True,
                  labels = False).astype(int)
pts_per_bin = df.groupby(bin_idxs).size()

num_var_wts = df['binary_out'] / pts_per_bin[bin_idxs].values

# plot the data using the calculated weights
plt.hist(data = df, x = 'num_var', bins = bin_edges, weights = num_var_wts)
plt.xlabel('num_var')
plt.ylabel('mean(binary_out)')
```

- get the mean of the y-variable ("binary_out") in each bin, the weight of each point should be equal to the y-variable value, divided by the number of points in its x-bin (num_var_wts). 
- we make use of pandas' cut function in order to associate each data point to a particular bin (bin_idxs)
- The labels = False parameter means that each point's bin membership is associated by a numeric index, rather than a string. We use these numeric indices into the pts_per_bin, with the .values at the end necessary in order for the Series' indices to not be confused between the indices of df['binary_out'].


# 13 - Line Plots
Plot trend of one numeric variable against values of a second variable
  - contrast with scatterplot where all data points are plotted
  - line plot only one point is plotted for each unique x value/bin of xvalues
    - if there are multiple values at that point, the y value is mean/median of the data in the bin

```
# set bin edges, compute centers
bin_size = 0.25
xbin_edges = np.arange(0.5, df['num_var1'].max()+bin_size, bin_size)
xbin_centers = (xbin_edges + bin_size/2)[:-1]

# compute statistics in each bin
data_xbins = pd.cut(df['num_var1'], xbin_edges, right = False, include_lowest = True)
y_means = df['num_var2'].groupby(data_xbins).mean()
y_sems = df['num_var2'].groupby(data_xbins).sem()

# plot the summarized data
plt.errorbar(x = xbin_centers, y = y_means, yerr = y_sems)
plt.xlabel('num_var1')
plt.ylabel('num_var2')
```

Alternate Variation
- make computations on rolling window through use of `rolling` (instead of fixed bins)

```
# compute statistics in a rolling window
df_window = df.sort_values('num_var1').rolling(15)
x_winmean = df_window.mean()['num_var1']
y_median = df_window.median()['num_var2']
y_q1 = df_window.quantile(.25)['num_var2']
y_q3 = df_window.quantile(.75)['num_var2']

# plot the summarized data
base_color = sb.color_palette()[0]
line_color = sb.color_palette('dark')[0]
plt.scatter(data = df, x = 'num_var1', y = 'num_var2')
plt.errorbar(x = x_winmean, y = y_median, c = line_color)
plt.errorbar(x = x_winmean, y = y_q1, c = line_color, linestyle = '--')
plt.errorbar(x = x_winmean, y = y_q3, c = line_color, linestyle = '--')

plt.xlabel('num_var1')
plt.ylabel('num_var2')
```

```
bin_edges = np.arange(-3, df['num_var'].max()+1/3, 1/3)
g = sb.FacetGrid(data = df, hue = 'cat_var', size = 5)
g.map(plt.hist, "num_var", bins = bin_edges, histtype = 'step')
g.add_legend()
```
  - performing the multiple hist calls through the use of FacetGrid, setting the categorical variable on the "hue" parameter rather than the "col" parameter
  - results in chart with lines stacked on top of each other with different colors

# 14 - Additional Plot Practice
Plot #1 where you only include makers that have at least 80 rows of data
  - start with data set up
    - get the frequency of the column `fuel_econ['make'].value_counts()`


# 15 - Lesson Summary
Learned a lot of different plot types.

Learned is used loosely here. I don't think I could make one of these on my own without a lot of struggle.

Hopefully my final project can use simple charts.

# 16 -  Postscript: Multivariate Visualization
Using color for third variate
```
plt.scatter(data = df, x = 'num_var1', y = 'num_var2', c = 'num_var3')
plt.colorbar()
```

Set different colors for different levels of categorical variable through the "hue" parameter on `FacetGrid` class

```
g = sb.FacetGrid(data = df, hue = 'cat_var1', size = 5)
g.map(plt.scatter, 'num_var1', 'num_var2')
g.add_legend()
```

Choose color palettes carefully as colors in sequence could lead people to believe the colors are meaningful (e.g. light blue to dark blue vs random cardinal colors)

FacetGrid allows you to facet across two variables

```
g = sb.FacetGrid(data = df, col = 'cat_var2', row = 'cat_var1', size = 2.5,
                margin_titles = True)
g.map(plt.scatter, 'num_var1', 'num_var2')
```

# 17 - Extra: Swarm Plots
Swarm plot is similar to a scatter plot
  - points are placed as close to their actual value as possible without allowing overlap

```
plt.figure(figsize = [12, 5])
base_color = sb.color_palette()[0]

# left plot: violin plot
plt.subplot(1, 3, 1)
ax1 = sb.violinplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)

# center plot: box plot
plt.subplot(1, 3, 2)
sb.boxplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)
plt.ylim(ax1.get_ylim()) # set y-axis limits to be same as left plot

# right plot: swarm plot
plt.subplot(1, 3, 3)
sb.swarmplot(data = df, x = 'cat_var', y = 'num_var', color = base_color)
plt.ylim(ax1.get_ylim()) # set y-axis limits to be same as left plot
```

Only use this if you have a small or moderate amount of data.

# 18 - Extra: Rug and Strip Plots

Rug Plot

```
g = sb.JointGrid(data = df, x = 'num_var1', y = 'num_var2')
g.plot_joint(plt.scatter)
g.plot_marginals(sb.rugplot, height = 0.25)
```

Strip Plot

```
plt.figure(figsize = [10, 5])
base_color = sb.color_palette()[0]

# left plot: strip plot
plt.subplot(1, 2, 1)
ax1 = sb.stripplot(data = df, x = 'num_var', y = 'cat_var',
                   color = base_color)

# right plot: violin plot with inner strip plot as lines
plt.subplot(1, 2, 2)
sb.violinplot(data = df, x = 'num_var', y = 'cat_var', color = base_color,
             inner = 'stick')
```

# 19 - Extra: Stacked Plots

Stacked Bar Charts

```
# pre-processing: count and sort by the number of instances of each category
sorted_counts = df['cat_var'].value_counts()

# establish the Figure
plt.figure(figsize = [12, 5])

# left plot: pie chart
plt.subplot(1, 2, 1)
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False);
plt.axis('square');

# right plot: horizontally stacked bar
plt.subplot(1, 2, 2)
baseline = 0
for i in range(sorted_counts.shape[0]):
    plt.barh(y = 1, width = sorted_counts[i], left = baseline)
    baseline += sorted_counts[i]

plt.legend(sorted_counts.index)  # add a legend for labeling
plt.ylim([0,2]) # give some vertical spacing around the bar
```

More interesting stacked bar chart with multiple things being counted:

```
cat1_order = ['East', 'South', 'West', 'North']
cat2_order = ['Type X', 'Type Y', 'Type Z', 'Type O']

plt.figure(figsize = [12, 5])

# left plot: clustered bar chart, absolute counts
plt.subplot(1, 2, 1)
sb.countplot(data = df, x = 'cat_var1', hue = 'cat_var2',
             order = cat1_order, hue_order = cat2_order)
plt.legend()

# right plot: stacked bar chart, absolute counts
plt.subplot(1, 2, 2)

baselines = np.zeros(len(cat1_order))
# for each second-variable category:
for i in range(len(cat2_order)):
    # isolate the counts of the first category,
    cat2 = cat2_order[i]
    inner_counts = df[df['cat_var2'] == cat2]['cat_var1'].value_counts()
    # then plot those counts on top of the accumulated baseline
    plt.bar(x = np.arange(len(cat1_order)), height = inner_counts[cat1_order],
            bottom = baselines)
    baselines += inner_counts[cat1_order]

plt.xticks(np.arange(len(cat1_order)), cat1_order)
plt.legend(cat2_order)
```

