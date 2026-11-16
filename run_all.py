import thermal_gen
import analyzer
import report_chart

print('--- thermal_gen ---')
t = thermal_gen.generate_thermal()
thermal_gen.np.save('sample_thermal.npy', t)
thermal_gen.cv2.imwrite('sample_thermal.png', thermal_gen.to_colormap(t))
print(f'min={t.min():.1f} max={t.max():.1f} mean={t.mean():.1f}')

print('--- analyzer ---')
analyzer.main()

print('--- report_chart ---')
report_chart.main()
