def ft_reduce(function_to_apply, list_of_inputs):
	tab=[]
	if (type(list_of_inputs) == set):
		list = []
		for i in list_of_inputs:
			list.append(i)
		for i in range(len(list) -1):
			tab.append(function_to_apply(list[i], list[i + 1]))
		return tab
	else:
		for i in range(len(list_of_inputs) - 1):
			tab.append(function_to_apply(list_of_inputs[i], list_of_inputs[i + 1]))
		return tab
