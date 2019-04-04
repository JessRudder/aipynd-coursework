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

# 10 - Categorical Plot Practice


# 11 - Faceting


# 12 - Adaptation of Univariate Plots


# 13 - Line Plots


# 14 - Additional Plot Practice


# 15 - Lesson Summary


# 16 -  Postscript: Multivariate Visualization


# 17 - Extra: Swarm Plots


# 18 - Extra: Rug and Strip Plots


# 19 - Extra: Stacked Plots

