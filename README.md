代码运行说明（简单版）

执行顺序
- data_preprocess → feature_select → 3_2 → stability_analysis
- .py 与 .ipynb 版本等价，可任选其一运行。

两种运行方式
- 使用 Jupyter Notebook：
  - 依次打开并运行对应的 .ipynb 文件（data_preprocess.ipynb → feature_select.ipynb → 3_2.ipynb → stability_analysis_3_0.ipynb）。
- 使用 Python 脚本：
  - 在本目录（code）下依次执行：
    - `python .\data_preprocess.py`
    - `python .\feature_select.py`
    - `python .\3_2.py`
    - `python .\stability_analysis_3_0.py`

注意事项
- 输入/输出路径以脚本内设置为准，如需调整请修改相应脚本中的路径配置。