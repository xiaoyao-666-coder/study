# Python脚本目录整理设计

日期：2026-07-15

## 1. 背景

当前 `s2s_rtist_source` 根目录共有140个Python脚本。Git当前正式跟踪的根级脚本只有9个，其余大多是尚未纳入版本管理的历史实验、数据准备、诊断、训练、评估和可视化脚本。根目录还存在9个 `.md/.txt` 文件，包括正式服务器说明、历史运行笔记、论文文本摘录、早期复现记录和依赖清单。

这些脚本长期按实验阶段直接堆放在根目录，存在以下问题：

- 很难判断哪个脚本是当前正式版本；
- 同一功能存在多个 `v1/v2/v3` 文件，替代关系不清楚；
- 正式流程、一次性诊断和历史复现脚本混在一起；
- 多个脚本按根目录模块名互相导入，直接移动会破坏运行；
- 服务器命令和文档依赖具体文件名，重命名后容易失效；
- 解释文档和依赖清单与脚本混放，文档版本和对应流程不清楚；
- 新对话或新成员需要全文搜索才能找到入口。

## 2. 已确认目标

1. 根目录最终只保留一个Python文件：`project_cli.py`。
2. 正式可复用逻辑进入标准Python包 `src/s2s_rtist/`。
3. 当前运行脚本按用途移动到 `scripts/` 分类目录。
4. 所有历史脚本保留并移动，不删除任何Python脚本。
5. 允许同步修改导入路径、测试、文档和服务器命令。
6. 增加可搜索的脚本目录和每个分类的Markdown说明。
7. 统一入口支持快速列出、搜索、查看和运行脚本。
8. 项目外现有未提交文件不修改、不暂存、不提交。
9. 根目录原有 `.md/.txt` 文件同步迁移并保留，不删除；解释文档进入 `docs/`，依赖清单进入 `requirements/`。
10. 统一入口的搜索范围同时覆盖脚本和文档。

## 3. 采用方案

采用“正式代码包化、当前脚本分类、历史脚本归档”的混合方案。

不采用仅增加索引而不移动文件的方案，因为它不能解决根目录拥挤问题。

不采用一次性把全部140个脚本重写为包模块的方案，因为历史实验脚本的依赖和数据条件差异很大，全面重构会增加复现风险。

## 4. 目标目录

```text
s2s_rtist_source/
|-- project_cli.py
|-- pyproject.toml
|-- src/
|   `-- s2s_rtist/
|       |-- __init__.py
|       |-- weather/
|       |-- physics/
|       |-- labels/
|       |-- validation/
|       `-- pipelines/
|-- scripts/
|   |-- README.md
|   |-- script_catalog.csv
|   |-- data_preparation/
|   |-- simulation/
|   |-- diagnostics/
|   |-- training/
|   |-- evaluation/
|   |-- visualization/
|   `-- archive/
|       |-- README.md
|       |-- VERSIONS.md
|       |-- superseded/
|       |-- one_off/
|       `-- original_application/
|-- requirements/
|   `-- requirements_gefs_gridmet_bias_validation_v1.txt
|-- tests/
`-- docs/
    |-- README.md
    |-- document_catalog.csv
    |-- operations/
    |   `-- server/
    |-- research/
    |   `-- reproduction/
    `-- archive/
        |-- README.md
        |-- historical_notes/
        `-- paper_extracts/
```

每个 `scripts/` 分类目录包含一个 `README.md`，说明目录职责、当前推荐脚本、输入输出、版本关系、统一入口ID和已知依赖。

`docs/README.md` 提供文档总导航，`docs/document_catalog.csv` 记录原文件名、当前路径、文档类型、状态、对应脚本和正式引用资格。

## 5. 正式包职责

### 5.1 `weather`

负责GEFS、gridMET、天气时间映射、累计量重构和天气偏差计算。

首批迁移来源：

- `gefs_gridmet_bias_validation_v1.py`

### 5.2 `physics`

负责根区或固定控制体储水、边界通量积分、移动边界项和水量平衡。

首批迁移来源：

- `rootzone_flux_frequency_diagnostic_v1.py`

### 5.3 `labels`

负责从SWAP输出提取正式代理模型标签。

首批迁移来源：

- `swap_three_output_labels_v1.py`

### 5.4 `validation`

负责数据契约、smoke输出、候选完整性和物理恒等式验证。

首批迁移来源：

- `validate_three_output_smoke_v1.py`

### 5.5 `pipelines`

负责可复用的数据生成编排，不包含一次性服务器启动逻辑。

首批迁移来源：

- `generate_restart_decision_dataset.py`

正式包内模块不再使用版本号文件名。版本信息由Git提交、数据处理规范和输出元数据管理。

## 6. 当前运行脚本分类

分类首先依据脚本职责，其次参考文件名前缀：

| 分类 | 主要前缀或职责 |
|---|---|
| `data_preparation` | `apply_`、`build_`、`extract_`、`merge_`、`prepare_`、`plan_` |
| `simulation` | SWAP运行、候选生成、restart编排、`decision_`、正式数据生成runner |
| `diagnostics` | `analyze_`、`audit_`、`compare_`、`diagnose_`、物理诊断runner |
| `training` | `train_`、`calibrate_`、`sweep_`、`resweep_`、训练期优化 |
| `evaluation` | `evaluate_`、`summarize_`、`collect_`、策略结果汇总 |
| `visualization` | `plot_`、`visualize_` |

文件名前缀只用于初始分类。最终分类以脚本实际行为为准，例如运行根区频率实验的runner应进入 `diagnostics`，而不是仅因为前缀为 `run_` 就进入 `simulation`。

## 7. 历史脚本和版本规则

所有历史脚本保留，不删除。

状态值固定为：

- `formal`：当前正式流程或正式包模块；
- `active`：仍用于当前研究但尚未正式锁定；
- `historical`：用于已结束实验的可复现脚本；
- `superseded`：已被明确的新版本替代；
- `legacy_unreviewed`：用途或替代关系暂时不能可靠判断。

归档规则：

1. 同一脚本存在多个版本时，只有在文档、依赖关系或结果目录能够证明替代关系后，旧版本才进入 `archive/superseded/`。
2. 不能只根据版本号最高就认定正式版本。
3. 一次性复现、临时检查和单案例脚本进入 `archive/one_off/`。
4. 原论文或原应用入口，例如历史 `Main_win*.py`，进入 `archive/original_application/`。
5. 用途不确定的脚本不擅自判废，按功能移动并标记 `legacy_unreviewed`。
6. 归档脚本默认不改算法，只处理移动后必需的导入兼容。

`scripts/archive/VERSIONS.md` 记录每个版本族的当前版本、旧版本、替代依据和相关结果文档。

## 8. 脚本目录

`scripts/script_catalog.csv` 是机器可读的唯一脚本目录，至少包含以下字段：

```text
id
original_path
current_path
category
status
purpose
replaced_by
formal_reference
source_sha256
current_sha256
```

约束：

- 每个迁移前脚本有且只有一条记录；
- `id` 和 `current_path` 全局唯一；
- `original_path` 永久保留用于追溯；
- `replaced_by` 只在替代关系有证据时填写；
- `source_sha256` 记录移动前内容；
- 未修改内容的脚本必须满足 `source_sha256 == current_sha256`；
- 因导入调整而修改的脚本必须在目录README中说明。

`docs/document_catalog.csv` 是机器可读的文档目录，至少包含以下字段：

```text
id
original_path
current_path
document_type
status
purpose
related_script_ids
formal_reference
source_sha256
current_sha256
```

脚本和文档catalog使用不同ID命名空间，但统一入口的搜索结果必须显示记录类型，避免同名混淆。

## 9. 统一入口

根目录只保留 `project_cli.py`，支持：

```text
python project_cli.py list
python project_cli.py list --type docs
python project_cli.py find rootzone
python project_cli.py show rootzone-frequency
python project_cli.py run rootzone-frequency -- <原脚本参数>
```

### 9.1 `list`

列出ID、记录类型、分类、状态和一句话用途。默认优先显示正式和当前脚本，可通过 `--type scripts|docs|all` 及状态参数筛选。

### 9.2 `find`

同时在脚本和文档catalog的ID、原文件名、当前路径、用途、状态、关联关系和正式引用中进行不区分大小写的关键词搜索。

### 9.3 `show`

显示一条脚本或文档记录的完整目录信息。脚本记录显示推荐命令、版本关系和README位置；文档记录显示对应脚本、正式引用资格和当前路径。

### 9.4 `run`

通过独立Python子进程运行目标脚本：

- 当前工作目录固定为项目根目录；
- 自动把 `src/`、目标脚本目录和必要的兼容目录加入 `PYTHONPATH`；
- `--` 后的参数原样传递；
- 子进程退出码原样返回；
- 未知ID返回非零退出码并给出相近ID建议；
- catalog标记为不可运行或缺依赖时，先显示原因，不伪装成已验证。

CLI是迁移后的正式运行方式。历史脚本仍可按新路径直接运行，但直接运行不作为主要兼容承诺。

## 10. 导入和服务器兼容

正式脚本统一使用 `s2s_rtist.*` 包导入，不再从项目根目录导入具体文件名。

CLI负责在无需安装包的情况下添加 `src/`。`pyproject.toml` 同时允许开发环境执行可编辑安装，但服务器运行不强制安装。

正式服务器说明、运行包和命令更新为 `project_cli.py` 稳定ID。服务器包必须保留以下相对结构：

```text
project_cli.py
src/s2s_rtist/
scripts/<required category>/
scripts/script_catalog.csv
```

历史脚本若依赖同目录模块，CLI把目标目录放在搜索路径首位。跨分类导入优先改为正式包导入；无法安全判断时，在catalog中标记依赖限制，不进行未经验证的算法重写。

## 11. Markdown导航

新增以下导航：

- `scripts/README.md`：总览、分类说明、正式入口和常用CLI示例；
- 每个分类目录的 `README.md`：脚本用途、输入输出、状态和版本；
- `scripts/archive/README.md`：归档规则和复现注意事项；
- `scripts/archive/VERSIONS.md`：版本替代关系；
- 正式包各子目录可使用简短README说明模块职责和公开接口。
- `docs/README.md`：文档总导航、正式引用入口和历史资料说明；
- `docs/operations/server/README.md`：服务器运行说明与对应CLI命令；
- `docs/research/reproduction/README.md`：复现记录及其数据/脚本依赖；
- `docs/archive/README.md`：历史说明和论文文本摘录的保留规则。

README和catalog必须一致。自动测试以catalog为结构事实来源，README用于人类导航。

### 11.1 根级文档迁移

根目录现有9个 `.md/.txt` 文件按以下规则迁移：

| 原文件 | 目标分类 |
|---|---|
| `fixed_0_100cm_5site_smoke_server_run_20260715.md` | `docs/operations/server/` |
| `formal_npd24_5site_smoke_server_run_20260714.md` | `docs/operations/server/` |
| `gefs_gridmet_bias_validation_server_run_20260715.md` | `docs/operations/server/` |
| `first_step_reproduction_notes_2026-05-29.md` | `docs/research/reproduction/` |
| `server_restart_smoke_notes_2026-05-30.md` | `docs/archive/historical_notes/` |
| `Instructions.txt` | `docs/archive/historical_notes/` |
| `paper_keyword_snippets.txt` | `docs/archive/paper_extracts/` |
| `paper_text_2026ems.txt` | `docs/archive/paper_extracts/` |
| `requirements_gefs_gridmet_bias_validation_v1.txt` | `requirements/` |

论文全文提取和关键词摘录只作为本地研究资料保留，不自动标记为正式引用来源。依赖清单虽然使用 `.txt` 扩展名，但属于运行配置，因此进入 `requirements/` 而不是 `docs/`。

## 12. 迁移步骤

1. 生成迁移前脚本清单，记录140个根级Python文件的路径、大小和SHA256。
2. 建立 `src/s2s_rtist/`、`scripts/` 分类目录和README骨架。
3. 先迁移正式核心模块，更新正式导入和现有测试。
4. 迁移正式runner并注册稳定CLI ID。
5. 按职责迁移其余当前脚本。
6. 根据明确替代证据移动历史版本和一次性脚本。
7. 生成完整脚本catalog和版本说明。
8. 记录根目录9个 `.md/.txt` 文件的原路径、大小和SHA256，并迁移到已确认目录。
9. 生成文档catalog和文档导航，更新服务器命令、正式交叉引用和运行包清单。
10. 执行结构、语法、导入、单元测试、文档路径和CLI验证。
11. 对照迁移前清单确认没有脚本或文档丢失。

迁移过程中不执行删除命令。若目标路径冲突，停止该文件迁移并记录冲突，不覆盖任何文件。

## 13. 错误处理

- catalog出现重复ID或重复当前路径时，测试失败；
- catalog引用不存在文件时，测试失败；
- 迁移前脚本未出现在catalog时，测试失败；
- 目标目录已存在同名文件时，不覆盖并报告；
- 正式脚本存在无法解析的本地导入时，测试失败；
- 历史脚本缺少第三方依赖或旧数据时，在catalog和README中记录，不将其标记为正式可运行；
- 文档仍引用旧根级脚本命令时，检查失败并列出位置。
- 文档catalog出现重复ID、缺失路径或遗漏迁移前根级 `.md/.txt` 文件时，测试失败；
- 正式文档移动后存在失效的相对链接时，检查失败并列出来源文件和目标链接。

## 14. 测试和验收

### 14.1 结构检查

- 根目录只有 `project_cli.py` 一个Python文件；
- 迁移前140个脚本全部可通过catalog追溯；
- catalog的ID、当前路径和原始路径映射完整且唯一；
- 所有catalog路径实际存在；
- 未修改历史脚本的SHA256保持一致。
- 根目录不再保留迁移前的9个 `.md/.txt` 文件；
- 9个文件全部可通过文档catalog追溯，未修改文件的SHA256保持一致。

### 14.2 CLI检查

- `list`、`find`、`show` 对脚本和文档都正常工作；
- 正式ID可以定位并转发 `--help` 或等价的无副作用参数；
- 未知ID给出清晰错误和相近建议；
- 子进程退出码正确透传。

### 14.3 Python检查

- 所有移动后的Python文件通过 `py_compile`；
- 正式包的本地导入可解析；
- 当前68项GEFS、根区、标签和smoke测试全部通过；
- 新增CLI、catalog覆盖和目录结构测试。

### 14.4 文档检查

- 正式文档不再出现失效的根级脚本路径；
- 正式服务器命令使用CLI稳定ID；
- 每个分类目录存在README；
- 每个归档版本族在 `VERSIONS.md` 中可追溯。
- 根目录迁移的9个说明/配置文件全部出现在 `docs/document_catalog.csv`；
- 正式文档的相对链接和对应脚本ID有效。

### 14.5 Git范围检查

- 只提交 `s2s_rtist_source` 内本次整理相关文件；
- 不修改或提交 `D:/study` 根目录已有的6个用户修改文件；
- 大型结果目录、气象数据、模型文件和归档包继续保持忽略。

## 15. 完成定义

只有同时满足以下条件才算整理完成：

1. 根目录达到单一入口目标；
2. 140个原始脚本全部保留且可追溯；
3. 根目录原有9个 `.md/.txt` 文件全部保留且可追溯；
4. 正式流程导入和CLI入口通过验证；
5. 现有68项测试和新增结构测试全部通过；
6. 正式文档、相对链接和服务器命令已更新；
7. 每个分类和版本关系都有Markdown说明；
8. 未将缺数据或缺依赖的历史脚本误报为已运行验证。
