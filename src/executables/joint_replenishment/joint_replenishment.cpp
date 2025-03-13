#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/vargroup.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/retrievestate.h"
#include "dynaplex/models/joint_replenishment/mdp.h"

using namespace DynaPlex;

int train() {

    for (int64_t k = 0; k < 1; k++) { // mdp configuration loop

        // init
        auto& dp = DynaPlexProvider::Get();
        std::string model_name = "joint_replenishment"; // define model name
        std::string mdp_config_name = "mdp_config_" + std::to_string(k) + ".json"; // add model configurations
        std::string mdp_config_name_testdata = "mdp_config_1.json"; // set up test demand data (change if more mdp environments are tested)

        // define mdp training environment
        std::string file_path = dp.System().filepath("mdp_config_examples", model_name, mdp_config_name);
        auto mdp_vars_from_json = DynaPlex::VarGroup::LoadFromFile(file_path);
        auto mdp = dp.GetMDP(mdp_vars_from_json);

        // define policy used to train neural network
        auto initialPolicy = mdp->GetPolicy("random");

        // set DCL parameters
        DynaPlex::VarGroup nn_training{
                {"early_stopping_patience",10},
                // { "max_training_epochs", 200 }
        };

        DynaPlex::VarGroup nn_architecture{
                {"type","mlp"},
                {"hidden_layers",DynaPlex::VarGroup::Int64Vec{128,128}}
        };

        int64_t num_gens = 4;
        DynaPlex::VarGroup dcl_config{
            {"N",20000}, // number of samples
            {"num_gens",num_gens},
            {"M",2000}, // number of exogenous samples
            {"H", 500},// rollout horizon length
            {"L", 0},// warm-up period
            {"nn_architecture",nn_architecture},
            {"nn_training",nn_training},
            {"retrain_lastgen_only",false}
            // {"reinitiate_counter", 200} // ensures not stuck in state
        };

        try {
            // train policy
            auto dcl = dp.GetDCL(mdp, initialPolicy, dcl_config);
            dcl.TrainPolicy(); // saves to disk

            // get all trained policies, as well as the initial policy, in vector form (generations)
            auto policies = dcl.GetPolicies();

            for (int64_t k = 0; k < 2; k++) {

                // retrieve policy to evaluate
                std::string policy_config_name = "policy_config_" + std::to_string(k) + ".json"; // 1 = can order 2 = periodic review
                std::string policy_file_path = dp.System().filepath("mdp_config_examples", model_name, policy_config_name);
                VarGroup evaluatePolicy = VarGroup::LoadFromFile(policy_file_path);
                auto policy = mdp->GetPolicy(evaluatePolicy);
                policies.push_back(policy);
            }

            // define test environment
            std::string file_path = dp.System().filepath("mdp_config_examples", model_name, mdp_config_name_testdata);
            mdp_vars_from_json = DynaPlex::VarGroup::LoadFromFile(file_path);
            auto mdp_test = dp.GetMDP(mdp_vars_from_json);

            // define comparison parameters
            VarGroup comparerConfig;
            comparerConfig.Add("number_of_trajectories", 1000);
            comparerConfig.Add("periods_per_trajectory", 1040);

            // compare policies
            auto comparer = dp.GetPolicyComparer(mdp_test, comparerConfig);
            auto comparison = comparer.Compare(policies);
            for (auto& VarGroup : comparison) {
                std::cout << VarGroup.Dump() << std::endl;
            }
        }

        catch (const std::exception& e) {
            std::cout << "exception: " << e.what() << std::endl;
        }
    }

    return 0;
}

int evaluate() {

    try {
        auto& dp = DynaPlexProvider::Get();
        DynaPlex::VarGroup config; // may not need
        auto& system = dp.System();
        std::string model_name = "joint_replenishment";
        std::string mdp_config_name = "mdp_config_0.json"; // define the mdp configuration you want to evaluate here
        std::string file_path = system.filepath("mdp_config_examples", model_name, mdp_config_name);

        // define mdp environment
        auto mdp_vars_from_json = VarGroup::LoadFromFile(file_path);
        auto mdp = dp.GetMDP(mdp_vars_from_json);
        using underlying_mdp = DynaPlex::Models::joint_replenishment::MDP;
        std::int64_t rng_seed = 11112014;
        Trajectory trajectory{ };
        trajectory.RNGProvider.SeedEventStreams(true, rng_seed);
        int64_t sampleNumber = 10400; // define sample horizon

        // get trained DCL policy
        std::string gen = "3"; // change to correct gen
        auto filename = dp.System().filepath(mdp->Identifier(), "dcl_policy_gen" + gen); // DynaPlex_IO folder
        auto path = dp.System().filepath(filename);
        auto dclPolicy = dp.LoadPolicy(mdp, path);
        
        // initiate state and state policies to evaluate
        std::vector<std::string> evaluatePolicyList = {"canOrderPolicy", "periodicReviewPolicy" };
        
        for (int64_t k = 0; k < evaluatePolicyList.size(); k++) {

            // initiate state
            mdp->InitiateState({ &trajectory,1 });

            // retrieve policy to evaluate
            std::string policy_config_name = "policy_config_" + std::to_string(k) + ".json"; // 1 = can order 2 = periodic review
            std::string policy_file_path = dp.System().filepath("mdp_config_examples", model_name, policy_config_name);
            VarGroup evaluatePolicy = VarGroup::LoadFromFile(policy_file_path);
            auto policy = mdp->GetPolicy(evaluatePolicy);

            // import data from mdp setup
            int64_t containerVolume;
            int64_t penaltyCost;
            int64_t nrProducts;
            int64_t leadTime;

            mdp_vars_from_json.Get("capacity", containerVolume);
            mdp_vars_from_json.Get("penaltyCost", penaltyCost);
            mdp_vars_from_json.Get("nrProducts", nrProducts);
            mdp_vars_from_json.Get("leadTime", leadTime);

            // init tracking KPIs
            float serviceLevel = 0;
            float fillRate = 0;
            float periodicity = 0;
            float testSL;
            int64_t backorderCost = 0;
            int64_t holdingCost = 0;
            int64_t orderingCost = 0;
            int64_t unmetDemand = 0;
            int64_t demand = 0;
            int64_t totalDemand = 0;
            std::vector<int64_t> inventoryLevelPrior(nrProducts, 0);
            std::vector<int64_t> backorderPrior(nrProducts, 0);
            std::vector<int64_t> backorderPost(nrProducts, 0);
            std::vector<int64_t> orderQtyPrior(nrProducts, 0);
            bool final_reached = false;

            // track and calculate KPIs
            for (int64_t i = 0; i < sampleNumber;) {

                auto& cat = trajectory.Category;
                auto& state = DynaPlex::RetrieveState<underlying_mdp::State>(trajectory.GetState());

                if (cat.IsAwaitEvent()) {
                    fillRate += state.usedCapacity / containerVolume; // later divide by number of orders

                    if (state.usedCapacity != 0) {
                        periodicity += 1; // here we define # orders; later divide by number of sample periods to get periodicity
                    }

                    for (int64_t j = 0; j < nrProducts; j++) { // store inventory position level before demand
                        auto& product = state.SKUs[j];

                        inventoryLevelPrior[j] = product.inventoryLevel;
                        orderQtyPrior[j] = product.orderQty[0]; // when event is incorporated, we lose this information
                    }

                    mdp->IncorporateEvent({ &trajectory,1 });

                    orderingCost += state.periodOrderingCosts;
                    holdingCost += state.periodHoldingCosts;
                    backorderCost += state.periodBackorderCosts;

                    for (int64_t j = 0; j < nrProducts; j++) { // derive demand from previous and current inventory levels
                        auto& product = state.SKUs[j];

                        demand = inventoryLevelPrior[j] - (product.inventoryLevel - orderQtyPrior[j]);
                        totalDemand += demand;

                        if (demand != 0) {
                            if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] > 0) {
                                unmetDemand += demand - inventoryLevelPrior[j];
                            }
                            else if (inventoryLevelPrior[j] < demand && inventoryLevelPrior[j] <= 0) {
                                unmetDemand += demand;
                            }
                        }
                    }
                    i++; // count number of sample periods
                }

                else if (cat.IsAwaitAction()) {
                    policy->SetAction({ &trajectory,1 });
                    mdp->IncorporateAction({ &trajectory,1 });
                }

                else if (cat.IsFinal()) {
                    final_reached = true;
                }
            }

            // determine final values
            holdingCost = holdingCost / sampleNumber;
            backorderCost = backorderCost / sampleNumber;
            orderingCost = orderingCost / sampleNumber;
            fillRate = fillRate / periodicity; // periodicity currently used as order counter
            periodicity = periodicity / sampleNumber;
            serviceLevel = (static_cast<float>(totalDemand) - unmetDemand) / totalDemand; // change

            std::cout << "Total demand: " << totalDemand << std::endl;
            std::cout << "Total demand met: " << totalDemand - unmetDemand << std::endl;

            // store variables after evaluation
            VarGroup kpis{};
            kpis.Add("serviceLevel", serviceLevel);
            kpis.Add("fillRate", fillRate);
            kpis.Add("periodicity", periodicity);
            kpis.Add("backorderCost", backorderCost);
            kpis.Add("holdingCost", holdingCost);
            kpis.Add("orderingCost", orderingCost);
            kpis.Add("totalCost", trajectory.CumulativeReturn / sampleNumber);

            // output evaluation to a file
            auto filename = dp.System().filepath("joint_replenishment", "Evaluation", evaluatePolicyList[k] + "_evaluation.json");
            kpis.SaveToFile(filename);
        }
    }

    catch (const DynaPlex::Error& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

int main() {

    train();
    // evaluate();
};
