# GUI 设计规范（团队统一标准）

> 适用范围：本机/团队所有新老工作区、所有新开发软件的界面设计。
> 风格来源：ssstoken AI 网关仪表盘（Tailwind 体系）+ Tauri 现代桌面感。
> 技术无关：本规范同时给出 **PySide6 (QSS)** 与 **Web (HTML/CSS)** 两套实现对照，
> 无论用哪个技术栈都按同一套设计令牌与组件规范实现。

---

## 一、设计语言

- **亮色为主基调**，暗色模式可选（同一套令牌切换，不改变布局）
- **青绿 teal 主色** `#14b8a6`（业务品牌色可替换，其余令牌不变）
- 圆角体系：**12px 控件 / 16px 卡片 / 999px 胶囊**；页面背景 `#f9fafb`，卡片纯白 + 极轻阴影
- 侧边栏导航布局：左侧 220px，内容区卡片化；交互元素 0.2s 过渡、容器 0.3s
- 禁用渐变滥用，仅主按钮与进度条使用横向渐变；其余靠留白与层次

## 二、设计令牌（Light / Dark）

| 令牌 | 亮色 | 暗色 | 用途 |
|---|---|---|---|
| `bg` | `#f9fafb` | `#0f172a` | 页面背景 |
| `card` | `#ffffff` | `#1e293b` | 卡片/输入框底 |
| `border` | `#e5e7eb` | `#334155` | 组件边框 |
| `border_soft` | `#f3f4f6` | `#2b3949` | 卡片内细分隔线 |
| `text` | `#111827` | `#e2e8f0` | 主文字 |
| `text_sec` | `#4b5563` | `#cbd5e1` | 次要文字 |
| `text_mut` | `#9ca3af` | `#94a3b8` | 弱化/占位文字 |
| `primary` | `#14b8a6` | `#2dd4bf` | 主色（hover `#0d9488`） |
| `primary_soft` | `#f0fdfa` | `rgba(19,78,74,.45)` | 主色浅底（选中态/标签） |
| `primary_text` | `#ffffff` | `#042f2e` | 主按钮文字 |
| `success` | `#10b981`（底 `#d1fae5` 字 `#047857`） | `#34d399`（底 18% 透明） | 成功徽章 |
| `warning` | `#f59e0b`（底 `#fef3c7` 字 `#b45309`） | `#fbbf24` | 警告徽章 |
| `danger` | `#ef4444`（底 `#fee2e2` 字 `#b91c1c`） | `#f87171` | 危险操作/错误徽章 |
| `hover` | `#f3f4f6` | `#243244` | 悬停底 |
| `shadow` | `rgba(0,0,0,.07)` 模糊 14px 偏移 y2 | 加深 | 卡片阴影 |

**圆角**：按钮/输入框/导航项/toast = `12px`；卡片/弹窗 = `16px`；小按钮 = `8px`；徽章/开关 = 全圆。
**字体**：`"Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif`；默认 13-14px、标题 20px/700、小节标题 14px/600；等宽（日志/数字）用 `Consolas, "Courier New", monospace` 12px。
**间距梯度**：4 / 8 / 12 / 14 / 16 / 20 / 24；卡片内边距 `16px 14px`，页面内容区 `20px 18px`，卡片间距 12-16。

## 三、组件规范

### 1. 按钮
| 变体 | 样式 |
|---|---|
| 主按钮 primary | teal 横向渐变 `#14b8a6→#0d9488`、白字、600、圆角 12、padding 8×16；hover 加深 |
| 次按钮 | 白底、1px `#e5e7eb` 边框、主文字；hover 底 `#f9fafb` 边框加深 |
| 幽灵 ghost | 透明底、次要文字；hover 底 `#f3f4f6` |
| 危险 danger | 红底白字，同主按钮结构 |
| 禁用 | 透明度 0.5、禁用 hover |
| 图标按钮 | 仅图标，同主按钮圆角 |

### 2. 卡片 `QFrame#card`
白底 / 16px 圆角 / 1px `#f3f4f6` 边框 / 轻阴影（QSS 无 box-shadow，用 `QGraphicsDropShadowEffect` 模糊 14 偏移 y2 透明度 0.07）；卡片标题用 `h2`（14px/600）。

### 3. 输入控件（QLineEdit/QSpinBox/QComboBox）
白底、1px 边框、圆角 12、padding 7×12；focus 边框变主色；下拉项圆角 10、选中主色浅底；文本选中主色底白字。

### 4. 徽章（胶囊）
`border-radius:11px`、padding 3×12、12px/600；状态色见令牌表（成功/警告/错误/中性=hover 底）。

### 5. 侧边栏导航
宽 220px、白底（暗色 `#0d1526`）、右分隔线；导航项圆角 12、padding 10×14、14px 左对齐；
**选中态** = 主色浅底 + 主色深字 600；hover = `#f3f4f6`。分组标题 12px 弱化字。

### 6. 标签页 QTabBar
页签圆角 10、padding 7×16；选中 = 主色浅底 + 主色字 600；容器无边框透明。

### 7. 表格
表头底 `#f9fafb` 字 `#4b5563` 500、单元格 padding 12×16、行分隔线 `#f3f4f6`、行 hover 底 `#f9fafb`。

### 8. 进度条
高 10px、圆角 6、teal 横向渐变块；文字居中 11px 次要色。

### 9. 滚动条
宽 10px、圆角 5、滑块 `#9ca3af`（hover 加深）、两端无箭头。

### 10. 弹窗 / Toast
弹窗卡片 16px 圆角、大阴影；遮罩 `rgba(0,0,0,.5)`；标题 18px/600，header/footer 分隔线 `#e5e7eb`。
Toast：右上角、圆角 12、左侧 4px 状态色条、入场右滑 0.3s（Qt 用 QPropertyAnimation）。

## 四、布局规范

- 主窗口：左侧边栏 220px + 右侧 QStackedWidget 页面
- 页面结构：标题行（20px 标题 + 13px 副标题 + 右侧状态徽章）→ 功能卡片组
- 卡片内部：小节标题 14px/600 → 表单/控件（间距 10）
- 日志/进度区放页面底部卡片，等宽字体
- 统一入口页导航最多 5 项；第 4 页起用标签页细分

## 五、主题切换

- 亮/暗两套令牌，切换时整体替换 QSS（`app.setStyleSheet`），过渡 0.25s
- 选择持久化到 `config/ui.json`（key `dark`）
- 主色高亮暗色提亮为 `#2dd4bf`，背景 `#0f172a`，卡片 `#1e293b`，边框 `#334155`
- 桌面应用可额外提供无边框窗口（自绘标题栏 + 拖拽区）获得 Tauri 式现代感

## 六、技术实现指南

### PySide6（QSS）
- 现成实现：`theme.py`（见本目录 `theme_pyside6.py` 副本，或漫画翻译助手项目）
- QSS 不支持：`box-shadow`（用 `QGraphicsDropShadowEffect`）、`backdrop-filter`（用半透明底色模拟）、`transform: scale`（用动画代替）
- 选中态切换：`setProperty("active", ...)` + `style().unpolish/polish`
- 列表/表格选中色：`selection-background-color: {primary}`

### Web（HTML/CSS）
- 现成样板：本目录 `样板-web.html`（浏览器打开即看；含无边框标题栏、侧边栏、卡片、亮暗切换）
- 建议配套：Tailwind CSS + shadcn/ui + Lucide 图标；CSS 变量按令牌表定义，`[data-theme]` 切换

## 七、接入清单（新项目）

1. 复制 `theme_pyside6.py`（Qt）或参考 `样板-web.html`（Web）
2. 替换令牌中的 `primary` 色为业务主色（其余不动）
3. 按「布局规范」搭骨架：侧边栏 + 页面卡片
4. 组件一律用规范样式，禁止临时写死颜色/圆角
5. 提供亮/暗切换（默认记忆上次选择）

---

*维护：本文件与两个实现样板为唯一权威来源；修改需同步更新全部文件。*
