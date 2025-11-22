import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Настройка стиля
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(16, 13))
ax.set_xlim(0, 16)
ax.set_ylim(0, 15)
ax.axis('off')

# Цвета
color_int8_packed = '#B2DFDB'
color_extract = '#FFF9C4'
color_int4_unsigned = '#FFE0B2'
color_int4_signed = '#D1C4E9'
color_fp16_act = '#E3F2FD'
color_fp16_weight = '#C8E6C9'
color_matmul = '#F8BBD0'
color_result = '#FFCDD2'
color_arrow = '#546E7A'

# Заголовок
ax.text(8, 14.2, 'INT4 Unpacking and Matrix Multiplication in Triton Kernel', 
        ha='center', va='top', fontsize=16, weight='bold')

# ========== ШАГ 1: Упакованные INT8 веса ==========
y_start = 12.5
ax.text(8, y_start + 0.5, 'Step 1: Packed INT8 Weights (from memory)', ha='center', fontsize=12, weight='bold')

# Пример упакованных значений (в hex для наглядности)
packed_values = [0x73, 0x52, 0x19, 0x6F]  # Каждый байт содержит 2 INT4
x_start = 4.5
box_width = 2.0
box_height = 0.7

for i, val in enumerate(packed_values):
    rect = FancyBboxPatch((x_start + i * box_width, y_start - 0.35), box_width * 0.95, box_height,
                          boxstyle="round,pad=0.05", edgecolor='#00796B', 
                          facecolor=color_int8_packed, linewidth=2.5)
    ax.add_patch(rect)
    ax.text(x_start + i * box_width + box_width/2, y_start + 0.15, f'0x{val:02X}',
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_start + i * box_width + box_width/2, y_start - 0.15, f'{val:08b}',
            ha='center', va='center', fontsize=8, family='monospace', color='#004D40')
    ax.text(x_start + i * box_width + box_width/2, y_start - 0.75, f'w_byte[{i}]',
            ha='center', va='top', fontsize=8, color='gray')

# ========== ШАГ 2: Извлечение Low/High Nibbles ==========
y_step2 = 10.0
ax.text(8, y_step2 + 0.8, 'Step 2: Extract Low & High Nibbles', ha='center', fontsize=12, weight='bold')

# Код операции
code_box = FancyBboxPatch((2, y_step2 + 0.15), 12, 0.45,
                         boxstyle="round,pad=0.08", edgecolor='#F57F17', 
                         facecolor=color_extract, linewidth=1.5, linestyle='--')
ax.add_patch(code_box)
ax.text(8, y_step2 + 0.37, 'low = w_byte & 0xF    |    high = (w_byte >> 4) & 0xF',
        ha='center', va='center', fontsize=9, family='monospace', weight='bold', color='#F57F00')

# Распакованные значения
ax.text(1.5, y_step2 - 0.3, 'High (bits 7-4):', ha='left', fontsize=10, weight='bold', color='#E65100')
ax.text(1.5, y_step2 - 1.2, 'Low (bits 3-0):', ha='left', fontsize=10, weight='bold', color='#1B5E20')

box_width_small = 1.3
x_nibble = 5

for i, val in enumerate(packed_values):
    high_nib = (val >> 4) & 0xF
    low_nib = val & 0xF
    
    # High nibble
    rect = FancyBboxPatch((x_nibble + i * box_width_small, y_step2 - 0.55), box_width_small * 0.95, 0.5,
                          boxstyle="round,pad=0.05", edgecolor='#E65100', 
                          facecolor=color_int4_unsigned, linewidth=2)
    ax.add_patch(rect)
    ax.text(x_nibble + i * box_width_small + box_width_small/2, y_step2 - 0.3, 
            f'{high_nib}', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x_nibble + i * box_width_small + box_width_small/2, y_step2 - 0.5, 
            f'({high_nib:04b})', ha='center', va='center', fontsize=7, family='monospace')
    
    # Low nibble
    rect = FancyBboxPatch((x_nibble + i * box_width_small, y_step2 - 1.45), box_width_small * 0.95, 0.5,
                          boxstyle="round,pad=0.05", edgecolor='#1B5E20', 
                          facecolor=color_int4_unsigned, linewidth=2)
    ax.add_patch(rect)
    ax.text(x_nibble + i * box_width_small + box_width_small/2, y_step2 - 1.2, 
            f'{low_nib}', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x_nibble + i * box_width_small + box_width_small/2, y_step2 - 1.4, 
            f'({low_nib:04b})', ha='center', va='center', fontsize=7, family='monospace')

# ========== ШАГ 3: Преобразование в Signed INT4 ==========
y_step3 = 7.5
ax.text(8, y_step3 + 0.8, 'Step 3: Convert to Signed INT4 [-8, 7]', ha='center', fontsize=12, weight='bold')

# Формула
formula_box = FancyBboxPatch((3, y_step3 + 0.15), 10, 0.45,
                            boxstyle="round,pad=0.08", edgecolor='#6A1B9A', 
                            facecolor=color_int4_signed, linewidth=1.5, linestyle='--')
ax.add_patch(formula_box)
ax.text(8, y_step3 + 0.37, 'signed = (value < 8) ? value : value - 16',
        ha='center', va='center', fontsize=9, family='monospace', weight='bold', color='#4A148C')

# Преобразованные значения
x_signed = 5
all_nibbles = []
for val in packed_values:
    high_nib = (val >> 4) & 0xF
    low_nib = val & 0xF
    all_nibbles.extend([low_nib, high_nib])

for i, nib in enumerate(all_nibbles):
    signed_val = nib if nib < 8 else nib - 16
    
    rect = FancyBboxPatch((x_signed + i * box_width_small, y_step3 - 0.5), box_width_small * 0.95, 0.6,
                          boxstyle="round,pad=0.05", edgecolor='#7B1FA2', 
                          facecolor=color_int4_signed, linewidth=2)
    ax.add_patch(rect)
    
    # Показываем преобразование
    ax.text(x_signed + i * box_width_small + box_width_small/2, y_step3 - 0.1, 
            f'{nib}→{signed_val:+d}', ha='center', va='center', fontsize=8, weight='bold')
    ax.text(x_signed + i * box_width_small + box_width_small/2, y_step3 - 0.35, 
            f'INT4', ha='center', va='center', fontsize=7, style='italic', color='#4A148C')
    ax.text(x_signed + i * box_width_small + box_width_small/2, y_step3 - 0.85, 
            f'w[{i}]', ha='center', va='top', fontsize=7, color='gray')

# ========== ШАГ 4: Матричное умножение ==========
y_step4 = 5.0
ax.text(8, y_step4 + 0.8, 'Step 4: Matrix Multiplication (INT4 weights × FP16 activations)', 
        ha='center', fontsize=12, weight='bold')

# Activation (FP16)
ax.text(2.5, y_step4 + 0.2, 'Activations\n(FP16):', ha='center', fontsize=9, weight='bold', color='#1565C0')
act_values = [2.5, 1.8, -0.9, 3.1]
x_act = 1.5
for i, act in enumerate(act_values):
    rect = FancyBboxPatch((x_act, y_step4 - 0.5 - i * 0.5), 1.2, 0.4,
                          boxstyle="round,pad=0.05", edgecolor='#1976D2', 
                          facecolor=color_fp16_act, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x_act + 0.6, y_step4 - 0.3 - i * 0.5, f'{act:.1f}',
            ha='center', va='center', fontsize=8, weight='bold')
    ax.text(x_act - 0.3, y_step4 - 0.3 - i * 0.5, f'x[{i}]',
            ha='right', va='center', fontsize=7, color='gray')

# Weights (INT4 -> FP16)
ax.text(5.5, y_step4 + 0.2, 'Weights (INT4→FP16):', ha='left', fontsize=9, weight='bold', color='#2E7D32')
x_weight = 5
for i in range(4):
    nib = all_nibbles[i]
    signed_val = nib if nib < 8 else nib - 16
    
    rect = FancyBboxPatch((x_weight + i * box_width_small, y_step4 - 0.3), box_width_small * 0.95, 0.4,
                          boxstyle="round,pad=0.05", edgecolor='#388E3C', 
                          facecolor=color_fp16_weight, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x_weight + i * box_width_small + box_width_small/2, y_step4 - 0.1, 
            f'{signed_val:+d}', ha='center', va='center', fontsize=8, weight='bold')

# Операция умножения
matmul_box = FancyBboxPatch((4.5, y_step4 - 1.3), 7, 0.6,
                           boxstyle="round,pad=0.1", edgecolor='#C2185B', 
                           facecolor=color_matmul, linewidth=2)
ax.add_patch(matmul_box)
ax.text(8, y_step4 - 1.0, 'acc += dot(activations_fp16, weights_fp16)',
        ha='center', va='center', fontsize=9, family='monospace', weight='bold', color='#880E4F')

# Стрелки к результату
ax.annotate('', xy=(4.3, y_step4 - 1.0), xytext=(2.7, y_step4 - 0.3),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#1976D2'))
ax.annotate('', xy=(6.5, y_step4 - 1.0), xytext=(6, y_step4 - 0.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#388E3C'))

# ========== ШАГ 5: Применение Scale и получение результата ==========
y_step5 = 2.5
ax.text(8, y_step5 + 0.8, 'Step 5: Apply Scale & Bias', ha='center', fontsize=12, weight='bold')

# Формула
scale_box = FancyBboxPatch((3, y_step5 + 0.1), 10, 0.5,
                          boxstyle="round,pad=0.08", edgecolor='#D32F2F', 
                          facecolor=color_result, linewidth=2)
ax.add_patch(scale_box)
ax.text(8, y_step5 + 0.45, 'result = (acc × scale / 7.0) + bias',
        ha='center', va='center', fontsize=10, family='monospace', weight='bold', color='#B71C1C')
ax.text(8, y_step5 + 0.23, 'scale = absmax of original FP16 weights',
        ha='center', va='center', fontsize=8, style='italic', color='#C62828')

# Финальный результат
result_box = FancyBboxPatch((5.5, y_step5 - 0.7), 5, 0.7,
                           boxstyle="round,pad=0.1", edgecolor='#D32F2F', 
                           facecolor=color_result, linewidth=2.5)
ax.add_patch(result_box)
ax.text(8, y_step5 - 0.35, 'Output: FP16 Result',
        ha='center', va='center', fontsize=11, weight='bold')

# ========== Стрелки между шагами ==========
arrow_props = dict(arrowstyle='->', lw=2.5, color=color_arrow)

# 1 -> 2
ax.annotate('', xy=(8, y_step2 + 0.7), xytext=(8, y_start - 1.2),
            arrowprops=arrow_props)

# 2 -> 3
ax.annotate('', xy=(8, y_step3 + 0.7), xytext=(8, y_step2 - 1.8),
            arrowprops=arrow_props)

# 3 -> 4
ax.annotate('', xy=(8, y_step4 + 0.7), xytext=(8, y_step3 - 1.2),
            arrowprops=arrow_props)

# 4 -> 5
ax.annotate('', xy=(8, y_step5 + 0.7), xytext=(8, y_step4 - 1.6),
            arrowprops=arrow_props)

# ========== Ключевые особенности ==========
info_y = 1.0
info_box = FancyBboxPatch((0.5, info_y - 0.3), 15, 0.8,
                         boxstyle="round,pad=0.1", edgecolor='#455A64', 
                         facecolor='#ECEFF1', linewidth=1.5, linestyle='--')
ax.add_patch(info_box)

ax.text(8, info_y + 0.25, '🚀 Key Features:', ha='center', va='center', 
        fontsize=10, weight='bold', color='#263238')
ax.text(8, info_y, '• Fused dequantization + matmul in single kernel (no intermediate FP16 weights stored)',
        ha='center', va='center', fontsize=8, style='italic')
ax.text(8, info_y - 0.25, '• Memory bandwidth: Load INT8 packed weights (2x less traffic vs FP16)',
        ha='center', va='center', fontsize=8, style='italic')

# Дополнительная информация про is_high selector
selector_note = FancyBboxPatch((11.5, y_step2 - 1.7), 3.8, 1.5,
                              boxstyle="round,pad=0.1", edgecolor='#FF6F00', 
                              facecolor='#FFF3E0', linewidth=1.5, linestyle=':')
ax.add_patch(selector_note)
ax.text(13.4, y_step2 - 0.65, 'Selector Logic:', ha='center', fontsize=8, weight='bold', color='#E65100')
ax.text(13.4, y_step2 - 0.95, 'is_high = (k_idx & 1)', ha='center', fontsize=7, 
        family='monospace', color='#BF360C')
ax.text(13.4, y_step2 - 1.2, 'w = is_high ? high : low', ha='center', fontsize=7, 
        family='monospace', color='#BF360C')
ax.text(13.4, y_step2 - 1.5, '(selects correct nibble', ha='center', fontsize=7, 
        style='italic', color='#E65100')
ax.text(13.4, y_step2 - 1.7, 'based on K dimension)', ha='center', fontsize=7, 
        style='italic', color='#E65100')

plt.tight_layout()
plt.savefig('dequantization_matmul_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Диаграмма деквантизации и матмула сохранена в 'dequantization_matmul_diagram.png'")
plt.show()

