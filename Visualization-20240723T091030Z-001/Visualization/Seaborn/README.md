# Seaborn Cheatsheet
0. **Install**: `pip install seaborn`
## General
1. **Import Seaborn**: `import seaborn as sns`
2. **Set default style**: `sns.set_style('whitegrid')`
3. **Set context**: `sns.set_context('notebook')`
4. **Set color palette**: `sns.set_palette('husl')`

## Data Loading
5. **Load example dataset**: `df = sns.load_dataset('tips')`
6. **Show available datasets**: `sns.get_dataset_names()`

## Plot Types

### Distribution Plots
7. **Histogram**: `sns.histplot(data=df, x='column')`
8. **Kernel Density Estimate (KDE) Plot**: `sns.kdeplot(data=df, x='column')`
9. **Rug Plot**: `sns.rugplot(data=df, x='column')`
10. **ECDF Plot**: `sns.ecdfplot(data=df, x='column')`

### Categorical Plots
11. **Bar Plot**: `sns.barplot(x='x', y='y', data=df)`
12. **Count Plot**: `sns.countplot(x='column', data=df)`
13. **Box Plot**: `sns.boxplot(x='x', y='y', data=df)`
14. **Violin Plot**: `sns.violinplot(x='x', y='y', data=df)`
15. **Swarm Plot**: `sns.swarmplot(x='x', y='y', data=df)`
16. **Strip Plot**: `sns.stripplot(x='x', y='y', data=df)`
17. **Point Plot**: `sns.pointplot(x='x', y='y', data=df)`
18. **Bar Plot with error bars**: `sns.barplot(x='x', y='y', data=df, ci='sd')`

### Regression Plots
19. **Regression Plot**: `sns.regplot(x='x', y='y', data=df)`
20. **LM Plot**: `sns.lmplot(x='x', y='y', data=df)`

### Relational Plots
21. **Scatter Plot**: `sns.scatterplot(x='x', y='y', data=df)`
22. **Line Plot**: `sns.lineplot(x='x', y='y', data=df)`

### Multi-plot Grids
23. **FacetGrid**: `g = sns.FacetGrid(df, col='column'); g.map(sns.histplot, 'data')`
24. **PairGrid**: `g = sns.PairGrid(df); g.map_lower(sns.scatterplot); g.map_diag(sns.histplot)`

### Heatmaps
25. **Heatmap**: `sns.heatmap(data=df.corr())`
26. **Clustered Heatmap**: `sns.clustermap(data=df)`

### Time Series Plots
27. **Time Series Line Plot**: `sns.lineplot(x='date', y='value', data=df)`

## Customizing Plots

### Axis Labels and Titles
28. **Set x-axis label**: `plt.xlabel('X-axis')`
29. **Set y-axis label**: `plt.ylabel('Y-axis')`
30. **Set plot title**: `plt.title('Title')`

### Legends
31. **Add legend**: `plt.legend(title='Legend')`

### Colors and Styles
32. **Set color palette**: `sns.set_palette('coolwarm')`
33. **Change plot color**: `sns.scatterplot(x='x', y='y', data=df, color='red')`
34. **Set plot style**: `sns.set_style('darkgrid')`

### Grid and Spines
35. **Remove grid**: `sns.despine()`
36. **Remove specific spines**: `sns.despine(top=True)`

### Annotations
37. **Annotate text**: `plt.text(x, y, 'text')`
38. **Annotate with arrow**: `plt.annotate('text', xy=(x, y), xytext=(x2, y2), arrowprops=dict(facecolor='black'))`

## Statistical Plots

### Residuals and Predictions
39. **Residual Plot**: `sns.residplot(x='x', y='y', data=df)`
40. **Confidence Interval**: `sns.pointplot(x='x', y='y', data=df, ci='sd')`

### Advanced Plots
41. **Pairplot**: `sns.pairplot(df)`
42. **Jointplot**: `sns.jointplot(x='x', y='y', data=df, kind='scatter')`
43. **Rugplot**: `sns.rugplot(data=df['x'])`
44. **FacetGrid with hue**: `g = sns.FacetGrid(df, col='column', hue='hue'); g.map(sns.histplot, 'data')`

## Plotting Techniques

### Line and Marker Styles
45. **Line plot with markers**: `sns.lineplot(x='x', y='y', data=df, marker='o')`
46. **Line plot with different line styles**: `sns.lineplot(x='x', y='y', data=df, linestyle='--')`

### Customizing Markers and Lines
47. **Scatter plot with different markers**: `sns.scatterplot(x='x', y='y', data=df, marker='x')`
48. **Scatter plot with size**: `sns.scatterplot(x='x', y='y', data=df, size='size_var')`

## Interactive Widgets
49. **Interactive plot with ipywidgets**: `from ipywidgets import interact; interact(function, param=value)`

## Data Wrangling for Plotting
50. **Melting DataFrame**: `df_melted = pd.melt(df, id_vars='id_vars', value_vars='value_vars')`
51. **Pivoting DataFrame**: `df_pivoted = df.pivot(index='index', columns='columns', values='values')`
52. **Grouping DataFrame**: `df_grouped = df.groupby('column').mean()`
53. **Filtering DataFrame**: `df_filtered = df[df['column'] > value]`

This should cover a broad range of Seaborn functionalities to help you with data visualization.
