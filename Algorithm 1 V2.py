import fly_path as fp

N = 3
charges = fp.generate_charges(N, False)
path_1, path_progress, forces, turn_point, peaks, og_peak_pos, og_peak_lines_1 = fp.fly_path((0, 0), (10, 0), charges, 1)
fp.plot_forces(forces, path_progress, peaks, turn_point)
path_2, path_progress, forces, turn_point, peaks, og_peak_pos, og_peak_lines_2 = fp.fly_path((10, 0), (0, 10), charges, 1)
fp.plot_forces(forces, path_progress, peaks, turn_point)
path_3, path_progress, forces, turn_point, peaks, og_peak_pos, og_peak_lines_3 = fp.fly_path((0, 10), (10, 10), charges, 1)
fp.plot_forces(forces, path_progress, peaks, turn_point)

paths = [path_1, path_2, path_3]
og_peak_list = [og_peak_lines_1, og_peak_lines_2, og_peak_lines_3]
fp.plot_field(N, charges, paths, og_peak_list)