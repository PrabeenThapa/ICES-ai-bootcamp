# Matplotlib Cheatsheet
0. **Install**: `pip install matplotlib`
## Basics
1. **Import Matplotlib**: `import matplotlib.pyplot as plt`
2. **Create a Figure**: `fig = plt.figure()`
3. **Add a subplot**: `ax = fig.add_subplot(111)`
4. **Basic plot**: `plt.plot(x, y)`
5. **Show plot**: `plt.show()`
6. **Save plot**: `plt.savefig('filename.png')`
7. **Set title**: `plt.title('Title')`
8. **Set x-label**: `plt.xlabel('X-axis label')`
9. **Set y-label**: `plt.ylabel('Y-axis label')`
10. **Set x-limits**: `plt.xlim([xmin, xmax])`
11. **Set y-limits**: `plt.ylim([ymin, ymax])`
12. **Set x-ticks**: `plt.xticks(ticks)`
13. **Set y-ticks**: `plt.yticks(ticks)`
14. **Grid on**: `plt.grid(True)`
15. **Grid off**: `plt.grid(False)`
16. **Log scale x-axis**: `plt.xscale('log')`
17. **Log scale y-axis**: `plt.yscale('log')`
18. **Set aspect ratio**: `plt.gca().set_aspect('equal')`
19. **Add legend**: `plt.legend()`
20. **Set line style**: `plt.plot(x, y, linestyle='--')`
21. **Set line color**: `plt.plot(x, y, color='red')`
22. **Set line width**: `plt.plot(x, y, linewidth=2)`
23. **Set marker style**: `plt.plot(x, y, marker='o')`
24. **Set marker size**: `plt.plot(x, y, markersize=10)`
25. **Set marker color**: `plt.plot(x, y, markerfacecolor='blue')`

## Plot Types
26. **Scatter plot**: `plt.scatter(x, y)`
27. **Bar plot**: `plt.bar(x, height)`
28. **Horizontal bar plot**: `plt.barh(x, width)`
29. **Histogram**: `plt.hist(x, bins=10)`
30. **Pie chart**: `plt.pie(sizes, labels=labels)`
31. **Box plot**: `plt.boxplot(data)`
32. **Error bar plot**: `plt.errorbar(x, y, yerr=error)`
33. **Stem plot**: `plt.stem(x, y)`
34. **Step plot**: `plt.step(x, y)`
35. **Fill between**: `plt.fill_between(x, y1, y2)`
36. **Contour plot**: `plt.contour(X, Y, Z)`
37. **Contourf plot**: `plt.contourf(X, Y, Z)`
38. **Quiver plot**: `plt.quiver(X, Y, U, V)`
39. **Stream plot**: `plt.streamplot(X, Y, U, V)`
40. **Hexbin plot**: `plt.hexbin(x, y, gridsize=30)`
41. **Polar plot**: `plt.polar(theta, r)`
42. **3D plot**: `ax.plot3D(x, y, z)`
43. **3D scatter plot**: `ax.scatter3D(x, y, z)`
44. **3D bar plot**: `ax.bar3D(x, y, z, dx, dy, dz)`
45. **3D contour plot**: `ax.contour3D(X, Y, Z)`
46. **3D surface plot**: `ax.plot_surface(X, Y, Z)`
47. **3D wireframe plot**: `ax.plot_wireframe(X, Y, Z)`

## Customization
48. **Set figure size**: `plt.figure(figsize=(10, 6))`
49. **Set DPI**: `plt.figure(dpi=100)`
50. **Add text**: `plt.text(x, y, 'text')`
51. **Add annotation**: `plt.annotate('text', xy=(x, y), xytext=(x2, y2), arrowprops=dict(facecolor='black'))`
52. **Set line style**: `plt.plot(x, y, linestyle='-.')`
53. **Set multiple styles**: `plt.plot(x, y, 'r--o')`
54. **Change background color**: `plt.gca().set_facecolor('lightgray')`
55. **Set grid style**: `plt.grid(color='gray', linestyle='--', linewidth=0.5)`
56. **Set tick parameters**: `plt.tick_params(axis='x', direction='in', length=5, width=2, colors='r')`
57. **Custom ticks**: `plt.xticks([1, 2, 3], ['A', 'B', 'C'])`
58. **Rotation of ticks**: `plt.xticks(rotation=45)`
59. **Hide ticks**: `plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)`
60. **Hide spines**: `plt.gca().spines['top'].set_visible(False)`
61. **Set spine color**: `plt.gca().spines['bottom'].set_color('blue')`
62. **Set spine line width**: `plt.gca().spines['left'].set_linewidth(2)`
63. **Adjust subplot spacing**: `plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)`
64. **Share x-axis**: `fig, (ax1, ax2) = plt.subplots(2, sharex=True)`
65. **Share y-axis**: `fig, (ax1, ax2) = plt.subplots(2, sharey=True)`
66. **Tight layout**: `plt.tight_layout()`

## Color
67. **Set color**: `plt.plot(x, y, color='green')`
68. **Set alpha (transparency)**: `plt.plot(x, y, alpha=0.5)`
69. **Colormap**: `plt.imshow(data, cmap='viridis')`
70. **List of colormaps**: `plt.colormaps()`
71. **Set color cycle**: `plt.gca().set_prop_cycle(color=['red', 'blue', 'green'])`
72. **Custom color cycle**: `from cycler import cycler; plt.gca().set_prop_cycle(cycler(color=['r', 'g', 'b'], linestyle=['-', '--', ':']))`

## Legends
73. **Add legend**: `plt.legend()`
74. **Legend location**: `plt.legend(loc='upper left')`
75. **Legend outside plot**: `plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`
76. **Set legend font size**: `plt.legend(fontsize='small')`
77. **Set legend title**: `plt.legend(title='Legend')`
78. **Remove legend frame**: `plt.legend(frameon=False)`
79. **Custom legend labels**: `plt.legend(['label1', 'label2'])`
80. **Multiple legend entries**: `plt.plot(x, y, label='Line 1'); plt.plot(x2, y2, label='Line 2'); plt.legend()`

## Advanced Plotting
81. **Twin x-axis**: `ax2 = ax.twinx()`
82. **Twin y-axis**: `ax2 = ax.twiny()`
83. **Secondary y-axis**: `ax.secondary_yaxis('right')`
84. **Secondary x-axis**: `ax.secondary_xaxis('top')`
85. **Inset plot**: `ax_inset = fig.add_axes([0.2, 0.5, 0.4, 0.4])`
86. **Broken axis**: `from brokenaxes import brokenaxes; bax = brokenaxes(ylims=((0, 1), (2, 3)))`
87. **Inset locator**: `from mpl_toolkits.axes_grid1.inset_locator import inset_axes; ax_inset = inset_axes(ax, width="30%", height="30%")`
88. **Multiple y-axes**: `fig, ax1 = plt.subplots(); ax2 = ax1.twinx()`
89. **Multiple x-axes**: `fig, ax1 = plt.subplots(); ax2 = ax1.twiny()`
90. **FacetGrid with seaborn**: `import seaborn as sns; g = sns.FacetGrid(df, col='column'); g.map(plt.scatter, 'x', 'y')`

## Subplots
91. **Create subplots**: `fig, axs = plt.subplots(2, 2)`
92. **Share x-axis subplots**: `fig, axs = plt.subplots(2, 2, sharex=True)`
93. **Share y-axis subplots**: `fig, axs = plt.subplots(2, 2, sharey=True)`
94. **Set subplot title**: `axs[0, 0].set_title('Title')`
95. **Set subplot xlabel**: `axs[0, 0].set_xlabel('X-axis')`
96. **Set subplot ylabel**: `axs[0, 0].set_ylabel('Y-axis')`
97. **Tight layout for subplots**: `plt.tight_layout()`
98. **Add space between subplots**: `fig.subplots_adjust(hspace=0.5, wspace=0.5)`
99. **Remove x-ticks in subplots**: `plt.setp(axs, xticks=[])`
100. **Remove y-ticks in subplots**: `plt.setp(axs, yticks=[])`

## 3D Plotting
101. **Import 3D toolkit**: `from mpl_toolkits.mplot3d import Axes3D`
102. **Create 3D axes**: `fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')`
103. **3D scatter plot**: `ax.scatter(x, y, z)`
104. **3D line plot**: `ax.plot(x, y, z)`
105. **3D wireframe**: `ax.plot_wireframe(X, Y, Z)`
106. **3D surface plot**: `ax.plot_surface(X, Y, Z)`
107. **3D contour plot**: `ax.contour3D(X, Y, Z)`
108. **3D bar plot**: `ax.bar3D(x, y, z, dx, dy, dz)`
109. **Set 3D labels**: `ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')`
110. **Set 3D limits**: `ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax]); ax.set_zlim([zmin, zmax])`

## Animation
111. **Import animation**: `from matplotlib.animation import FuncAnimation`
112. **Define animation function**: `def animate(i): pass`
113. **Create animation**: `ani = FuncAnimation(fig, animate, frames=100, interval=20, blit=True)`
114. **Save animation**: `ani.save('animation.gif', writer='imagemagick')`

## Styles
115. **Set style**: `plt.style.use('ggplot')`
116. **List available styles**: `plt.style.available`
117. **Temporarily set style**: `with plt.style.context('fivethirtyeight'): pass`
118. **Set seaborn style**: `import seaborn as sns; sns.set_style('whitegrid')`

## Colorbars
119. **Add colorbar**: `plt.colorbar()`
120. **Set colorbar label**: `cbar.set_label('Label')`
121. **Set colorbar ticks**: `cbar.set_ticks([0, 1, 2])`
122. **Set colorbar limits**: `cbar.set_clim(vmin, vmax)`

## Text and Annotations
123. **Add text**: `plt.text(x, y, 'text')`
124. **Add annotation**: `plt.annotate('text', xy=(x, y), xytext=(x2, y2), arrowprops=dict(facecolor='black', shrink=0.05))`
125. **Set text properties**: `plt.text(x, y, 'text', fontsize=12, color='red', style='italic', weight='bold')`

## Advanced Customization
126. **Set tick labels**: `ax.set_xticklabels(labels)`
127. **Set tick label size**: `ax.tick_params(axis='both', which='major', labelsize=10)`
128. **Rotate tick labels**: `plt.xticks(rotation=45)`
129. **Hide tick labels**: `plt.xticks([])`
130. **Hide axis**: `ax.axis('off')`
131. **Set equal aspect ratio**: `ax.set_aspect('equal', adjustable='box')`

## Plot Inset
132. **Create inset axes**: `from mpl_toolkits.axes_grid1.inset_locator import inset_axes; axins = inset_axes(ax, width='30%', height='30%', loc='upper right')`
133. **Plot on inset**: `axins.plot(x, y)`
134. **Zoom effect on inset**: `from mpl_toolkits.axes_grid1.inset_locator import mark_inset; mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5')`

## Data Handling
135. **Load data from CSV**: `import pandas as pd; data = pd.read_csv('file.csv')`
136. **Plot data from DataFrame**: `plt.plot(data['x'], data['y'])`
137. **Bar plot from DataFrame**: `data.plot.bar(x='x', y='y')`
138. **Histogram from DataFrame**: `data['column'].plot.hist()`
139. **Box plot from DataFrame**: `data.boxplot(column='column')`
140. **Scatter plot from DataFrame**: `data.plot.scatter(x='x', y='y')`

## Logarithmic Scale
141. **Set x-axis to log scale**: `plt.xscale('log')`
142. **Set y-axis to log scale**: `plt.yscale('log')`
143. **Set both axes to log scale**: `plt.yscale('log'); plt.xscale('log')`

## Date Handling
144. **Plot with dates**: `import pandas as pd; dates = pd.date_range('20210101', periods=100); plt.plot(dates, y)`
145. **Format date axis**: `import matplotlib.dates as mdates; plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))`
146. **Rotate date labels**: `plt.gcf().autofmt_xdate()`
147. **Set date locator**: `plt.gca().xaxis.set_major_locator(mdates.MonthLocator())`

## Image Handling
148. **Display image**: `img = plt.imread('file.png'); plt.imshow(img)`
149. **Save plot as image**: `plt.savefig('plot.png')`
150. **Set image interpolation**: `plt.imshow(img, interpolation='nearest')`
151. **Set image cmap**: `plt.imshow(img, cmap='gray')`

## Statistical Plots
152. **Box plot**: `plt.boxplot(data)`
153. **Violin plot**: `plt.violinplot(data)`
154. **Swarm plot**: `import seaborn as sns; sns.swarmplot(x='x', y='y', data=data)`
155. **Pair plot**: `sns.pairplot(data)`
156. **Heatmap**: `sns.heatmap(data.corr(), annot=True, cmap='coolwarm')`
157. **Joint plot**: `sns.jointplot(x='x', y='y', data=data, kind='reg')`
158. **Regression plot**: `sns.regplot(x='x', y='y', data=data)`
159. **Dist plot**: `sns.distplot(data['column'])`
160. **Facet Grid**: `g = sns.FacetGrid(data, col='column'); g.map(plt.hist, 'column')`

## Interactive Plots
161. **Enable interactive mode**: `plt.ion()`
162. **Disable interactive mode**: `plt.ioff()`
163. **Interactive plot**: `plt.draw()`
164. **Update plot**: `plt.pause(0.01)`

## Configurations
165. **Set default figure size**: `plt.rcParams['figure.figsize'] = [10, 6]`
166. **Set default DPI**: `plt.rcParams['figure.dpi'] = 100`
167. **Set default linewidth**: `plt.rcParams['lines.linewidth'] = 2`
168. **Set default color cycle**: `plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['red', 'green', 'blue'])`
169. **Set default font size**: `plt.rcParams['font.size'] = 12`

## Figure Properties
170. **Set figure face color**: `fig.patch.set_facecolor('gray')`
171. **Set figure edge color**: `fig.patch.set_edgecolor('blue')`
172. **Set figure title**: `fig.suptitle('Figure Title')`
173. **Set figure tight layout**: `fig.tight_layout()`

## Exporting
174. **Save figure**: `fig.savefig('filename.png')`
175. **Save figure with DPI**: `fig.savefig('filename.png', dpi=300)`
176. **Save figure as PDF**: `fig.savefig('filename.pdf')`
177. **Save figure as SVG**: `fig.savefig('filename.svg')`

## Advanced
178. **Custom color map**: `from matplotlib.colors import LinearSegmentedColormap; cmap = LinearSegmentedColormap.from_list('name', ['red', 'blue', 'green']); plt.imshow(data, cmap=cmap)`
179. **Multiple subplots**: `fig, axs = plt.subplots(nrows=2, ncols=3)`
180. **Shared colorbar**: `fig.colorbar(im, ax=axs.ravel().tolist())`
181. **Matplotlib widgets**: `from matplotlib.widgets import Slider; slider = Slider(ax, 'Label', valmin, valmax)`
182. **Toggle grid**: `plt.grid()`
183. **Show grid only for x-axis**: `plt.grid(axis='x')`
184. **Show grid only for y-axis**: `plt.grid(axis='y')`
185. **Set minor ticks**: `plt.minorticks_on()`
186. **Set minor ticks off**: `plt.minorticks_off()`
187. **Logit scale x-axis**: `plt.xscale('logit')`
188. **Logit scale y-axis**: `plt.yscale('logit')`

## Miscellaneous
189. **Display multiple plots**: `plt.figure(); plt.plot(x, y); plt.figure(); plt.plot(x2, y2)`
190. **Plot with specific axis**: `fig, ax = plt.subplots(); ax.plot(x, y)`
191. **Add horizontal line**: `plt.axhline(y=value, color='r', linestyle='--')`
192. **Add vertical line**: `plt.axvline(x=value, color='r', linestyle='--')`
193. **Add horizontal span**: `plt.axhspan(ymin, ymax, color='red', alpha=0.5)`
194. **Add vertical span**: `plt.axvspan(xmin, xmax, color='red', alpha=0.5)`
195. **Get current axes**: `ax = plt.gca()`
196. **Get current figure**: `fig = plt.gcf()`
197. **Close plot**: `plt.close()`
198. **Close all plots**: `plt.close('all')`
199. **Interactive mode on**: `plt.ion()`
200. **Interactive mode off**: `plt.ioff()`
201. **Draw plot**: `plt.draw()`
202. **Pause plot**: `plt.pause(0.1)`

## Advanced Usage
203. **Set axis ticks inside**: `plt.tick_params(axis='both', direction='in')`
204. **Set tick direction**: `plt.tick_params(axis='x', direction='inout')`
205. **Set tick length**: `plt.tick_params(axis='y', length=10)`
206. **Set tick width**: `plt.tick_params(axis='both', width=2)`
207. **Set tick colors**: `plt.tick_params(axis='x', colors='red')`
208. **Set tick label size**: `plt.tick_params(axis='both', labelsize='large')`
209. **Set major ticks**: `plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))`
210. **Set minor ticks**: `plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.2))`
211. **Customize major ticks**: `plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))`
212. **Customize minor ticks**: `plt.gca().xaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))`
213. **Hide major ticks**: `plt.tick_params(axis='x', which='major', bottom=False)`
214. **Hide minor ticks**: `plt.tick_params(axis='x', which='minor', bottom=False)`

## Plot Customization
215. **Set plot title font size**: `plt.title('Title', fontsize=15)`
216. **Set plot title font style**: `plt.title('Title', fontstyle='italic')`
217. **Set plot title font weight**: `plt.title('Title', fontweight='bold')`
218. **Set axis label font size**: `plt.xlabel('X-axis', fontsize=12)`
219. **Set axis label font style**: `plt.xlabel('X-axis', fontstyle='italic')`
220. **Set axis label font weight**: `plt.xlabel('X-axis', fontweight='bold')`
221. **Set tick label font size**: `plt.xticks(fontsize=10)`
222. **Set tick label font style**: `plt.xticks(fontstyle='italic')`
223. **Set tick label font weight**: `plt.xticks(fontweight='bold')`
224. **Set axis tick direction**: `plt.tick_params(axis='both', direction='in')`
225. **Set axis tick length**: `plt.tick_params(axis='both', length=6)`
226. **Set axis tick width**: `plt.tick_params(axis='both', width=2)`
227. **Set axis tick colors**: `plt.tick_params(axis='both', colors='blue')`
228. **Set axis grid style**: `plt.grid(linestyle='--')`
229. **Set axis grid color**: `plt.grid(color='gray')`
230. **Set axis grid line width**: `plt.grid(linewidth=0.5)`
231. **Set axis grid alpha**: `plt.grid(alpha=0.7)`

## Data Visualization
232. **Box plot with notches**: `plt.boxplot(data, notch=True)`
233. **Box plot with different colors**: `plt.boxplot(data, patch_artist=True)`
234. **Violin plot with split**: `sns.violinplot(x='x', y='y', hue='hue', data=data, split=True)`
235. **Heatmap with annotations**: `sns.heatmap(data, annot=True)`
236. **Pair plot with hue**: `sns.pairplot(data, hue='hue')`
237. **Joint plot with kind**: `sns.jointplot(x='x', y='y', data=data, kind='kde')`
238. **Dist plot with bins**: `sns.distplot(data['column'], bins=20)`
239. **Facet Grid with hue**: `g = sns.FacetGrid(data, col='col', hue='hue'); g.map(plt.scatter, 'x', 'y')`
240. **Bar plot with hue**: `sns.barplot(x='x', y='y', hue='hue', data=data)`
241. **Point plot**: `sns.pointplot(x='x', y='y', hue='hue', data=data)`
242. **Line plot with error bars**: `sns.lineplot(x='x', y='y', hue='hue', data=data, ci='sd')`

## Complex Plots
243. **Subplot with polar projection**: `fig.add_subplot(111, polar=True)`
244. **Hexbin plot with gridsize**: `plt.hexbin(x, y, gridsize=50)`
245. **Bivariate KDE plot**: `sns.kdeplot(x='x', y='y', data=data)`
246. **Marginal histograms**: `import seaborn as sns; sns.jointplot(x='x', y='y', data=data, kind='hex')`
247. **Custom colormap in hexbin**: `plt.hexbin(x, y, cmap='inferno')`
248. **3D scatter with color mapping**: `sc = ax.scatter(x, y, z, c=c, cmap='viridis')`
249. **Dynamic range color mapping**: `plt.imshow(data, cmap='viridis', vmin=0, vmax=100)`
250. **Multicolor line plot**: `plt.plot(x, y, 'r-', x2, y2, 'b--')`
