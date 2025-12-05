import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Rectangle, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Hot Air Flow of Direct Heating Drum Roaster", fontsize=18, weight='bold', pad=20)

# ===== 1. EXHAUST SYSTEM (Top Left) =====
# Exhaust pipe
exhaust_pipe = Rectangle((0.8, 8), 0.3, 1.5, facecolor='#8B4513', edgecolor='#654321', linewidth=2)
ax.add_patch(exhaust_pipe)
ax.text(0.95, 9.7, 'EXHAUST', fontsize=10, weight='bold', ha='center')

# Dust room chamber
dust_room = FancyBboxPatch((0.4, 6.5), 1.1, 1.3, boxstyle="round,pad=0.05", 
                           facecolor='#E8E8E8', edgecolor='#333333', linewidth=2)
ax.add_patch(dust_room)
ax.text(0.95, 7.5, 'DUST\nROOM', fontsize=9, weight='bold', ha='center', va='center')

# Chaff collector (cone shape)
chaff_cone_x = [0.5, 0.95, 1.4, 0.95]
chaff_cone_y = [5.2, 6.4, 5.2, 5.2]
ax.fill(chaff_cone_x, chaff_cone_y, facecolor='#D3D3D3', edgecolor='#333333', linewidth=2)
ax.add_patch(Circle((0.95, 4.5), 0.4, facecolor='#CCCCCC', edgecolor='#333333', linewidth=2))
ax.text(0.95, 4.0, 'CHAFF\n& DEBRIS', fontsize=8, weight='bold', ha='center', va='top')

# ===== 2. ROASTING DRUM (Center) =====
# Main drum body
drum_main = Rectangle((3.5, 5.5), 3.5, 2, facecolor='#F5F5F5', edgecolor='#333333', linewidth=3)
ax.add_patch(drum_main)

# Drum details - louvers/vents
for i in range(4):
    vent = Rectangle((3.7 + i*0.8, 5.7), 0.5, 0.3, facecolor='#CCCCCC', edgecolor='#666666', linewidth=1)
    ax.add_patch(vent)

# Hopper (funnel top)
hopper_x = [7.5, 8.5, 8.2, 7.8]
hopper_y = [7.5, 7.5, 8.5, 8.5]
ax.fill(hopper_x, hopper_y, facecolor='#E0E0E0', edgecolor='#333333', linewidth=2)

# Coffee beans visualization
theta = np.linspace(0, 2*np.pi, 100)
bean_x = 5.5 + 0.8 * np.cos(theta)
bean_y = 6.5 + 0.4 * np.sin(theta)
ax.fill(bean_x, bean_y, facecolor='#8B4513', edgecolor='#654321', linewidth=1, alpha=0.8)
ax.text(5.5, 6.5, 'COFFEE', fontsize=10, weight='bold', ha='center', va='center', color='white')

# Heat source (flames)
flame_positions = [(4.0, 5.0), (4.7, 5.0), (5.4, 5.0), (6.1, 5.0)]
for fx, fy in flame_positions:
    flame = Wedge((fx, fy), 0.15, 0, 180, facecolor='#FF4500', edgecolor='#FF0000', linewidth=1)
    ax.add_patch(flame)
    ax.text(fx, fy-0.1, '火', fontsize=12, ha='center', va='top', color='#FF0000', weight='bold')

# Motor/control box
motor_box = Rectangle((3.2, 4.5), 0.8, 0.9, facecolor='#A9A9A9', edgecolor='#333333', linewidth=2)
ax.add_patch(motor_box)

# Drum support
ax.plot([3.5, 3.5, 3.0], [5.5, 4.8, 4.8], 'k-', linewidth=3)
ax.plot([7.0, 7.0, 7.5], [5.5, 4.8, 4.8], 'k-', linewidth=3)

# ===== 3. COOLING TRAY (Bottom Right) =====
cooling_tray = Rectangle((8.5, 1.5), 2.5, 1.8, facecolor='#F0F0F0', edgecolor='#333333', linewidth=3)
ax.add_patch(cooling_tray)

# Cooling fan
fan_center = (9.3, 2.4)
ax.add_patch(Circle(fan_center, 0.3, facecolor='white', edgecolor='#333333', linewidth=2))
ax.plot([fan_center[0]-0.2, fan_center[0]+0.2], [fan_center[1], fan_center[1]], 'k-', linewidth=2)
ax.plot([fan_center[0], fan_center[0]], [fan_center[1]-0.2, fan_center[1]+0.2], 'k-', linewidth=2)

# Cooling tray perforations
for i in range(3):
    for j in range(2):
        ax.plot([8.7 + i*0.6, 9.1 + i*0.6], [1.7 + j*0.8, 1.7 + j*0.8], 
                color='#4682B4', linewidth=3, alpha=0.7)

ax.text(10.5, 2.5, 'COOLING TRAY', fontsize=10, weight='bold', ha='right', va='center')

# Wheels
wheel1 = Circle((8.7, 1.3), 0.15, facecolor='#333333', edgecolor='#000000', linewidth=2)
wheel2 = Circle((10.8, 1.3), 0.15, facecolor='#333333', edgecolor='#000000', linewidth=2)
ax.add_patch(wheel1)
ax.add_patch(wheel2)

# ===== 4. HOT AIR FLOW ARROWS =====
# Hot air flow path annotations
air_flow_paths = [
    # From heat source through drum
    FancyArrowPatch((2.5, 6.5), (3.3, 6.5), arrowstyle='->', mutation_scale=30, 
                    linewidth=3, color='#FF6347', linestyle='--', alpha=0.8),
    FancyArrowPatch((3.3, 6.8), (6.8, 6.8), arrowstyle='->', mutation_scale=30, 
                    linewidth=3, color='#FF6347', linestyle='--', alpha=0.8),
    # Circular flow inside drum
    FancyArrowPatch((4.5, 7.3), (5.5, 7.3), arrowstyle='->', mutation_scale=25, 
                    linewidth=2.5, color='#FF4500', alpha=0.7),
    FancyArrowPatch((6.5, 7.0), (6.5, 6.0), arrowstyle='->', mutation_scale=25, 
                    linewidth=2.5, color='#FF4500', alpha=0.7),
    FancyArrowPatch((5.5, 5.8), (4.5, 5.8), arrowstyle='->', mutation_scale=25, 
                    linewidth=2.5, color='#FF4500', alpha=0.7),
    # To dust room
    FancyArrowPatch((1.5, 7.1), (2.5, 7.1), arrowstyle='->', mutation_scale=25, 
                    linewidth=2.5, color='#FF6347', alpha=0.7),
    # Up to exhaust
    FancyArrowPatch((0.95, 7.8), (0.95, 8.5), arrowstyle='->', mutation_scale=25, 
                    linewidth=2.5, color='#DC143C', alpha=0.7),
    # Chaff down
    FancyArrowPatch((0.7, 6.4), (0.7, 5.5), arrowstyle='->', mutation_scale=20, 
                    linewidth=2, color='#8B4513', linestyle=':', alpha=0.7),
    FancyArrowPatch((1.2, 6.4), (1.2, 5.5), arrowstyle='->', mutation_scale=20, 
                    linewidth=2, color='#8B4513', linestyle=':', alpha=0.7),
    # To cooling tray
    FancyArrowPatch((7.2, 5.5), (8.3, 3.5), arrowstyle='->', mutation_scale=30, 
                    linewidth=3, color='#4682B4', alpha=0.7),
    # Cooling air
    FancyArrowPatch((8.6, 2.1), (9.0, 2.1), arrowstyle='->', mutation_scale=20, 
                    linewidth=2.5, color='#4682B4', alpha=0.7),
    FancyArrowPatch((9.6, 2.1), (10.0, 2.1), arrowstyle='->', mutation_scale=20, 
                    linewidth=2.5, color='#4682B4', alpha=0.7),
]

for arrow in air_flow_paths:
    ax.add_patch(arrow)

# ===== 5. LABELS AND ANNOTATIONS =====
# HOT AIR FLOW label
ax.text(2.5, 7.0, 'HOT AIR FLOW', fontsize=11, weight='bold', 
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE4B5', edgecolor='#FF6347', linewidth=2))

# Annotation boxes with descriptions
annotations = [
    (0.95, 3.2, 'Chaff and debris\ncollected here', '#F5DEB3'),
    (5.5, 4.7, 'Direct heating with\nhot air circulation', '#FFE4E1'),
    (9.7, 0.8, 'Rapid cooling to\npreserve flavor', '#E0F2F7'),
]

for x, y, text, bgcolor in annotations:
    ax.text(x, y, text, fontsize=9, ha='center', va='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=bgcolor, edgecolor='#666666', 
                     linewidth=1.5, alpha=0.9))

# Title annotation at bottom
ax.text(6, 0.3, 'Hot air flow of direct heating drum roaster', 
        fontsize=12, ha='center', style='italic', color='#333333')

# ===== 6. INTERACTIVE ANNOTATIONS =====
# Create clickable interactive annotations
from matplotlib.widgets import Button
from matplotlib.patches import ConnectionPatch

# Store annotation state
annotation_boxes = {}
current_annotation = {'box': None}

def show_annotation(event, component_name, description, position):
    """Show detailed annotation when clicking on a component"""
    if current_annotation['box'] is not None:
        current_annotation['box'].remove()
        current_annotation['box'] = None
    
    if event.inaxes == ax:
        # Create annotation box
        bbox_props = dict(boxstyle='round,pad=0.8', facecolor='#FFFACD', 
                         edgecolor='#FF6347', linewidth=3, alpha=0.95)
        annotation = ax.annotate(f'{component_name}\n\n{description}',
                                xy=position, xytext=(position[0]+1.5, position[1]+1),
                                fontsize=10, ha='left', va='center',
                                bbox=bbox_props,
                                arrowprops=dict(arrowstyle='->', lw=2, color='#FF6347'))
        current_annotation['box'] = annotation
        plt.draw()

# Make components interactive by adding invisible clickable regions
interactive_components = [
    {'name': 'Exhaust System', 'pos': (0.95, 8.5), 
     'desc': 'Releases smoke and maintains\nairflow and temperature control.\nRemoves volatile compounds.'},
    {'name': 'Dust Room', 'pos': (0.95, 7.1), 
     'desc': 'Separates chaff and fine particles\nfrom the airstream using cyclonic\nseparation principles.'},
    {'name': 'Roasting Drum', 'pos': (5.5, 6.5), 
     'desc': 'Rotating drum with hot air circulation.\nDirect heating method: 200-250°C.\nRotation prevents burning.'},
    {'name': 'Heat Source', 'pos': (5.0, 5.0), 
     'desc': 'Gas burners provide direct heat.\nTemperature controlled by\ngas flow and air intake.'},
    {'name': 'Cooling Tray', 'pos': (9.7, 2.4), 
     'desc': 'Rapidly cools roasted beans to\nhalt the roasting process.\nPreserves flavor profile.'},
    {'name': 'Chaff Collector', 'pos': (0.95, 4.5), 
     'desc': 'Collects the thin papery skin\nthat separates from beans\nduring roasting.'},
]

# Add invisible circles for click detection
for comp in interactive_components:
    circle = Circle(comp['pos'], 0.4, facecolor='red', alpha=0.01, picker=True)
    circle.component_info = comp
    ax.add_patch(circle)

def on_pick(event):
    """Handle click events on components"""
    if hasattr(event.artist, 'component_info'):
        comp = event.artist.component_info
        show_annotation(event, comp['name'], comp['desc'], comp['pos'])

fig.canvas.mpl_connect('pick_event', on_pick)

# Add instruction text
ax.text(6, 9.5, '🖱️ Click on any component for detailed information', 
        fontsize=11, ha='center', weight='bold',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#E6F3FF', 
                 edgecolor='#4169E1', linewidth=2))

# Save outputs
plt.tight_layout()
plt.savefig('visual_roaster_machine.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Interactive visualization saved as: visual_roaster_machine.png")
print(f"📊 Diagram shows complete hot air flow path:")
print(f"   1. Hot air enters through heat source")
print(f"   2. Circulates through rotating drum")
print(f"   3. Exits to dust room for particle separation")
print(f"   4. Chaff collected separately")
print(f"   5. Exhaust releases filtered air")
print(f"   6. Roasted beans move to cooling tray")
print(f"\n💡 Click on components to see detailed descriptions!")

plt.show()
