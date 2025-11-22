import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Настройка стиля
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
ax.axis('off')

# Цвета
color_fp16 = '#E3F2FD'
color_even = '#C8E6C9'
color_odd = '#FFE0B2'
color_scale = '#F8BBD0'
color_int4 = '#D1C4E9'
color_int8 = '#B2DFDB'
color_arrow = '#546E7A'

# Заголовок
ax.text(8, 13.2, 'INT4 Quantization with INT8 Packing (FP16 → INT4 → INT8)', 
        ha='center', va='top', fontsize=16, weight='bold')

# ========== ШАГ 1: Исходные FP16 данные ==========
y_start = 11.5
ax.text(8, y_start + 0.5, 'Step 1: Input FP16 Row', ha='center', fontsize=12, weight='bold')

# Пример значений
fp16_values = [3.2, -1.5, 2.8, -0.7, 1.9, -2.3, 0.5, -1.1]
x_start = 2.5
box_width = 1.2
box_height = 0.6

for i, val in enumerate(fp16_values):
    rect = FancyBboxPatch((x_start + i * box_width, y_start - 0.3), box_width * 0.95, box_height,
                          boxstyle="round,pad=0.05", edgecolor='#1976D2', 
                          facecolor=color_fp16, linewidth=2)
    ax.add_patch(rect)
    ax.text(x_start + i * box_width + box_width/2, y_start, f'{val}',
            ha='center', va='center', fontsize=9, weight='bold')
    # Индекс
    ax.text(x_start + i * box_width + box_width/2, y_start - 0.7, f'[{i}]',
            ha='center', va='top', fontsize=8, color='gray')

# ========== ШАГ 2: Разделение на Even/Odd ==========
y_step2 = 9.0
ax.text(8, y_step2 + 0.8, 'Step 2: Split into Even/Odd Positions', ha='center', fontsize=12, weight='bold')

# Even positions
ax.text(2, y_step2 + 0.3, 'Even (0, 2, 4, 6):', ha='left', fontsize=10, weight='bold', color='#388E3C')
even_vals = [fp16_values[i] for i in range(0, len(fp16_values), 2)]
x_even = 5
for i, val in enumerate(even_vals):
    rect = FancyBboxPatch((x_even + i * box_width, y_step2 - 0.2), box_width * 0.95, box_height,
                          boxstyle="round,pad=0.05", edgecolor='#388E3C', 
                          facecolor=color_even, linewidth=2)
    ax.add_patch(rect)
    ax.text(x_even + i * box_width + box_width/2, y_step2 + 0.1, f'{val}',
            ha='center', va='center', fontsize=9, weight='bold')

# Odd positions
ax.text(2, y_step2 - 0.8, 'Odd (1, 3, 5, 7):', ha='left', fontsize=10, weight='bold', color='#F57C00')
odd_vals = [fp16_values[i] for i in range(1, len(fp16_values), 2)]
x_odd = 5
for i, val in enumerate(odd_vals):
    rect = FancyBboxPatch((x_odd + i * box_width, y_step2 - 1.3), box_width * 0.95, box_height,
                          boxstyle="round,pad=0.05", edgecolor='#F57C00', 
                          facecolor=color_odd, linewidth=2)
    ax.add_patch(rect)
    ax.text(x_odd + i * box_width + box_width/2, y_step2 - 1.0, f'{val}',
            ha='center', va='center', fontsize=9, weight='bold')

# ========== ШАГ 3: Вычисление Scale ==========
y_step3 = 6.5
ax.text(8, y_step3 + 0.8, 'Step 3: Calculate Scale', ha='center', fontsize=12, weight='bold')

# Формула
absmax_val = max([abs(v) for v in fp16_values])
scale_val = 7.0 / absmax_val
formula_box = FancyBboxPatch((3, y_step3 - 0.3), 10, 0.8,
                            boxstyle="round,pad=0.1", edgecolor='#C2185B', 
                            facecolor=color_scale, linewidth=2)
ax.add_patch(formula_box)
ax.text(8, y_step3 + 0.3, f'absmax = max(|even|, |odd|) = {absmax_val:.1f}', 
        ha='center', va='center', fontsize=10, weight='bold')
ax.text(8, y_step3 - 0.05, f'scale = 7.0 / absmax = {scale_val:.3f}', 
        ha='center', va='center', fontsize=10, weight='bold')

# ========== ШАГ 4: Масштабирование и Округление ==========
y_step4 = 4.5
ax.text(8, y_step4 + 0.8, 'Step 4: Scale & Round to INT4 [-7, 7]', ha='center', fontsize=12, weight='bold')

# Scaled and quantized values
ax.text(2, y_step4 + 0.2, 'Scaled & Quantized:', ha='left', fontsize=10, weight='bold')
quantized_vals = []
for val in fp16_values:
    scaled = val * scale_val
    if scaled >= 0:
        q = int(scaled + 0.5)
    else:
        q = int(scaled - 0.5)
    q = max(-7, min(7, q))  # Clamp to INT4 range
    quantized_vals.append(q)

x_quant = 2.5
for i, q_val in enumerate(quantized_vals):
    rect = FancyBboxPatch((x_quant + i * box_width, y_step4 - 0.4), box_width * 0.95, box_height,
                          boxstyle="round,pad=0.05", edgecolor='#7B1FA2', 
                          facecolor=color_int4, linewidth=2)
    ax.add_patch(rect)
    ax.text(x_quant + i * box_width + box_width/2, y_step4 - 0.1, f'{q_val}',
            ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x_quant + i * box_width + box_width/2, y_step4 - 0.8, f'[{i}]',
            ha='center', va='top', fontsize=8, color='gray')

# ========== ШАГ 5: Упаковка в INT8 ==========
y_step5 = 2.0
ax.text(8, y_step5 + 0.8, 'Step 5: Pack into INT8 (2 x INT4 → 1 x INT8)', ha='center', fontsize=12, weight='bold')
ax.text(8, y_step5 + 0.4, 'packed = (odd << 4) | even', ha='center', fontsize=9, 
        style='italic', color='#00695C')

x_packed = 3.5
box_width_packed = 2.0
for i in range(0, len(quantized_vals), 2):
    even_q = quantized_vals[i] & 0xF
    if i + 1 < len(quantized_vals):
        odd_q = quantized_vals[i + 1] & 0xF
    else:
        odd_q = 0
    
    packed_val = (odd_q << 4) | even_q
    
    # Большой бокс для упакованного значения
    rect = FancyBboxPatch((x_packed + (i//2) * box_width_packed, y_step5 - 0.5), 
                          box_width_packed * 0.95, 0.8,
                          boxstyle="round,pad=0.08", edgecolor='#00796B', 
                          facecolor=color_int8, linewidth=3)
    ax.add_patch(rect)
    
    # Показываем биты
    ax.text(x_packed + (i//2) * box_width_packed + box_width_packed/4, y_step5 - 0.1, 
            f'{odd_q:04b}', ha='center', va='center', fontsize=8, 
            family='monospace', color='#F57C00', weight='bold')
    ax.text(x_packed + (i//2) * box_width_packed + 3*box_width_packed/4, y_step5 - 0.1, 
            f'{even_q:04b}', ha='center', va='center', fontsize=8, 
            family='monospace', color='#388E3C', weight='bold')
    
    # Hex значение
    ax.text(x_packed + (i//2) * box_width_packed + box_width_packed/2, y_step5 + 0.25, 
            f'0x{packed_val:02X}', ha='center', va='center', fontsize=10, weight='bold')
    
    # Индекс в output
    ax.text(x_packed + (i//2) * box_width_packed + box_width_packed/2, y_step5 - 0.85, 
            f'out[{i//2}]', ha='center', va='top', fontsize=8, color='gray')

# Подписи для битов
ax.text(x_packed + box_width_packed/4, y_step5 - 1.2, 'bits 7-4\n(odd)', 
        ha='center', va='top', fontsize=7, color='#F57C00', style='italic')
ax.text(x_packed + 3*box_width_packed/4, y_step5 - 1.2, 'bits 3-0\n(even)', 
        ha='center', va='top', fontsize=7, color='#388E3C', style='italic')

# ========== Стрелки между шагами ==========
arrow_props = dict(arrowstyle='->', lw=2.5, color=color_arrow)

# 1 -> 2
ax.annotate('', xy=(8, y_step2 + 0.6), xytext=(8, y_start - 1.2),
            arrowprops=arrow_props)

# 2 -> 3
ax.annotate('', xy=(8, y_step3 + 0.6), xytext=(8, y_step2 - 1.8),
            arrowprops=arrow_props)

# 3 -> 4
ax.annotate('', xy=(8, y_step4 + 0.6), xytext=(8, y_step3 - 0.8),
            arrowprops=arrow_props)

# 4 -> 5
ax.annotate('', xy=(8, y_step5 + 0.6), xytext=(8, y_step4 - 1.3),
            arrowprops=arrow_props)

# ========== Дополнительная информация ==========
info_y = 0.3
info_box = FancyBboxPatch((0.5, info_y - 0.2), 15, 0.5,
                         boxstyle="round,pad=0.1", edgecolor='#455A64', 
                         facecolor='#ECEFF1', linewidth=1.5, linestyle='--')
ax.add_patch(info_box)
ax.text(8, info_y + 0.15, '💾 Memory Savings: 8 x FP16 (16 bytes) → 4 x INT8 (4 bytes) + 1 x FP16 scale (2 bytes) = 67% compression', 
        ha='center', va='center', fontsize=9, style='italic', weight='bold')

plt.tight_layout()
plt.savefig('quantization_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Диаграмма сохранена в 'quantization_diagram.png'")
plt.show()

