# ============================================================
# Web Traffic Analytics
# Author: Zeenu
# Description: Data Cleaning, EDA, Visualization & Insights
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   WEB TRAFFIC ANALYTICS")
print("=" * 60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv('Web_Traffic_Data.csv')
print(f"\n[DATA LOADED] Shape: {df.shape}")
print(df.head())

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['DayOfWeek'] = df['Date'].dt.day_name()
df['SessionDuration_min'] = (df['SessionDuration_sec'] / 60).round(2)
print(f"\nCleaned Shape: {df.shape}")
df.to_csv('cleaned_traffic_data.csv', index=False)
print("[SAVED] cleaned_traffic_data.csv")

# ============================================================
# STEP 3: EDA
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

total_sessions   = len(df)
total_pageviews  = df['PageViews'].sum()
avg_duration     = df['SessionDuration_min'].mean()
bounce_rate      = (df['Bounced'].sum() / total_sessions) * 100
total_conversions= df['Conversions'].sum()
conversion_rate  = (total_conversions / total_sessions) * 100
new_users        = (df['NewUser'].sum() / total_sessions) * 100

print(f"\nTotal Sessions      : {total_sessions:,}")
print(f"Total Page Views    : {total_pageviews:,}")
print(f"Avg Session Duration: {avg_duration:.2f} minutes")
print(f"Bounce Rate         : {bounce_rate:.2f}%")
print(f"Conversion Rate     : {conversion_rate:.2f}%")
print(f"New Users           : {new_users:.2f}%")

print(f"\nTop Pages:\n{df['Page'].value_counts().head()}")
print(f"\nTop Traffic Sources:\n{df['TrafficSource'].value_counts()}")
print(f"\nDevice Breakdown:\n{df['Device'].value_counts()}")

# ============================================================
# STEP 4: VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: CREATING VISUALIZATIONS")
print("=" * 60)

sns.set_theme(style="whitegrid")
BLUE  = '#1E88E5'
RED   = '#E53935'
GREEN = '#43A047'
COLORS_PIE = ['#1E88E5','#E53935','#43A047','#FB8C00','#8E24AA','#00ACC1']

# --- Graph 1: Traffic Source (Pie) ---
src = df['TrafficSource'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(src.values, labels=src.index, autopct='%1.1f%%',
       colors=COLORS_PIE, startangle=140,
       wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('Traffic by Source', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('traffic_graphs/traffic_by_source.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/traffic_by_source.png")

# --- Graph 2: Top Pages (Bar) ---
pages = df['Page'].value_counts().head(8)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(pages.index, pages.values, color=BLUE, edgecolor='white', linewidth=1.5)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            str(bar.get_height()), ha='center', fontsize=9, fontweight='bold')
ax.set_title('Most Visited Pages', fontsize=14, fontweight='bold')
ax.set_xlabel('Page', fontsize=12)
ax.set_ylabel('Number of Sessions', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('traffic_graphs/top_pages.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/top_pages.png")

# --- Graph 3: Device Breakdown (Pie) ---
dev = df['Device'].value_counts()
fig, ax = plt.subplots(figsize=(7, 6))
ax.pie(dev.values, labels=dev.index, autopct='%1.1f%%',
       colors=[BLUE, RED, GREEN], startangle=90,
       wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('Sessions by Device Type', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('traffic_graphs/device_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/device_breakdown.png")

# --- Graph 4: Session Duration Histogram ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df['SessionDuration_min'], bins=40, color=BLUE, edgecolor='white', linewidth=0.8)
ax.axvline(avg_duration, color=RED, linestyle='--', linewidth=2, label=f'Avg: {avg_duration:.1f} min')
ax.set_title('Session Duration Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Session Duration (minutes)', fontsize=12)
ax.set_ylabel('Number of Sessions', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('traffic_graphs/session_duration.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/session_duration.png")

# --- Graph 5: Monthly Traffic Trend ---
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
monthly = df.groupby('Month').size().reindex(month_order).fillna(0)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly.index, monthly.values, color=BLUE, marker='o',
        linewidth=2.5, markersize=7)
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color=BLUE)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly.index, rotation=45, ha='right')
ax.set_title('Monthly Traffic Trend (2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Sessions', fontsize=12)
plt.tight_layout()
plt.savefig('traffic_graphs/monthly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/monthly_trend.png")

# --- Graph 6: Country Traffic (Bar) ---
country = df['Country'].value_counts().head(7)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(country.index[::-1], country.values[::-1],
               color=GREEN, edgecolor='white', linewidth=1.5)
for bar in bars:
    ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
            str(int(bar.get_width())), va='center', fontsize=9, fontweight='bold')
ax.set_title('Traffic by Country', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Sessions', fontsize=12)
plt.tight_layout()
plt.savefig('traffic_graphs/traffic_by_country.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/traffic_by_country.png")

# --- Graph 7: Bounce Rate by Page ---
bounce_page = df.groupby('Page')['Bounced'].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(10, 5))
colors_b = [RED if v > 30 else BLUE for v in bounce_page.values]
bars = ax.bar(bounce_page.index, bounce_page.values, color=colors_b, edgecolor='white')
for bar, val in zip(bars, bounce_page.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Bounce Rate by Page', fontsize=14, fontweight='bold')
ax.set_xlabel('Page', fontsize=12)
ax.set_ylabel('Bounce Rate (%)', fontsize=12)
ax.axhline(30, color='orange', linestyle='--', linewidth=1.5, label='30% threshold')
ax.legend()
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('traffic_graphs/bounce_rate_by_page.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/bounce_rate_by_page.png")

# --- Graph 8: New vs Returning Users ---
user_counts = df['NewUser'].value_counts()
labels = ['Returning Users', 'New Users']
fig, ax = plt.subplots(figsize=(7, 6))
ax.pie(user_counts.values, labels=labels, autopct='%1.1f%%',
       colors=[BLUE, GREEN], startangle=140,
       wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('New vs Returning Users', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('traffic_graphs/new_vs_returning.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] traffic_graphs/new_vs_returning.png")

print("\n" + "=" * 60)
print("  ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)
