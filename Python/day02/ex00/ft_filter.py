def ft_filter(function_to_apply, list_of_inputs):
	tab = []
	for elem in list_of_inputs:
		if (function_to_apply(elem)):
			tab.append(elem)
	return tab
