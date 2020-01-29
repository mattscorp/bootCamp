def ft_map(function_to_apply, list_of_inputs):
	tab = []
	for elem in list_of_inputs:
		tab.append(function_to_apply(elem))
	return tab
