#include <pybind11/pybind11.h>
#include "dynaplex/vargroup.h"
#include "vargroupcaster.h"
#include "dynaplex/error.h"
#include "dynaplex/neuralnetworktrainer.h"
#include "dynaplex/utilities.h"
#include <torch/torch.h>
#include <dynaplex/mdp.h>
#include <dynaplex/provider.h>

namespace py = pybind11;
/*
namespace DynaPlex {
	DynaPlex::MDP GetMDP(py::kwargs& kwargs)
	{
		auto vars = DynaPlex::VarGroup(kwargs);
		return DynaPlex::GetMDP(vars);
	}
}
*/
std::string TestParam(DynaPlex::VarGroup& vars)
{
	return vars.ToAbbrvString();
}

std::string TestParam(py::kwargs& kwargs)
{
	auto vars = DynaPlex::VarGroup(kwargs);
	return TestParam(vars );
}

void TestTorch()
{
	DynaPlex::NeuralNetworkTrainer trainer{};
	std::cout << trainer.TorchAvailability() << std::endl;
}

DynaPlex::VarGroup GetVarGroup()
{
	DynaPlex::VarGroup distprops{
			{"type","geom"},
		{"mean",5} };

	return distprops;
}


namespace DynaPlex {

	// Create a static Provider to manage registrations.
	static Provider s_provider;

	DynaPlex::MDP GetMDP(py::kwargs& kwargs) {
		auto vars = DynaPlex::VarGroup(kwargs);
		return s_provider.GetMDP(vars);  // Use s_provider to get the MDP
	}

	DynaPlex::MDP GetMDP(const DynaPlex::VarGroup& vars) {
		return s_provider.GetMDP(vars);  // Use s_provider to get the MDP
	}

	DynaPlex::VarGroup ListMDPs()
	{
		return s_provider.ListMDPs();
	}

}


	
PYBIND11_MODULE(DP_Bindings, m) {

	m.doc() = "DynaPlex extension for Python";
	m.def("get_var_group", &GetVarGroup, "gets some parameters");
	m.def("test_torch", &TestTorch, "tests pytorch availability");
	// Expose the MDPInterface
	py::class_<DynaPlex::MDPInterface,DynaPlex::MDP>(m, "MDP")
		.def("identifier", &DynaPlex::MDPInterface::Identifier);

	m.def("test_param", py::overload_cast<py::kwargs&>(&TestParam), "simply prints the named params. ");
	m.def("test_param", py::overload_cast<DynaPlex::VarGroup&>(&TestParam), "simply prints param. ");

	m.def("list_mdps", &DynaPlex::ListMDPs, "Lists available MDPs");
	m.def("get_mdp", py::overload_cast<py::kwargs&>(&DynaPlex::GetMDP), "Gets MDP based on dictionary.");
	m.def("get_mdp", py::overload_cast<const DynaPlex::VarGroup&>(&DynaPlex::GetMDP), "Gets MDP based on dictionary.");

}