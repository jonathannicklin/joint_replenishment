#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/vargroup.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/retrievestate.h"
#include "dynaplex/models/joint_replenishment/mdp.h"

using namespace DynaPlex;

void exportToCSV(const std::vector<std::vector<int64_t>>& data, const std::string& filename) {
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file is open
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    // Loop through each row and each column to write data into CSV format
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            // Write the value followed by a comma, but no comma after the last element in a row
            outFile << data[i][j];

            // Check if this is not the last column, then add a comma
            if (j != data[i].size() - 1) {
                outFile << ";";
            }
        }
        // After each row, add a new line
        outFile << "\n";
    }

    // Close the file after writing
    outFile.close();
}

int train() {

    for (int64_t k = 0; k < 1; k++) { // mdp configuration loop

        // init
        auto& dp = DynaPlexProvider::Get();
        std::string model_name = "joint_replenishment"; // define model name
        std::string mdp_config_name = "mdp_config_" + std::to_string(k) + ".json"; // add model configurations
        std::string mdp_config_name_testdata = "mdp_config_0.json"; // set up test demand data (change if more mdp environments are tested)

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
        Trajectory trajectory{};
        trajectory.RNGProvider.SeedEventStreams(true, rng_seed);
        int64_t sampleNumber = 10400; // define sample horizon

        bool ruleBasedEvaluate = false;
        bool dclEvaluate = true;
        bool ppoEvaluate = false;

        if (ruleBasedEvaluate == true) {

            // initiate state and state policies to evaluate
            // std::vector<std::string> evaluatePolicyList = { "canOrderPolicy", "periodicReviewPolicy" };
            std::vector<std::string> evaluatePolicyList = { "canOrderPolicy"};

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

                // containers for graphing
                std::vector<std::vector<int64_t>> inventoryPositionsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));
                std::vector<std::vector<int64_t>> inventoryLevelsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));

                // init tracking KPIs
                float serviceLevel = 0;
                float fillRate = 0;
                float periodicity = 0;
                float itemsPerOrder = 0;
                int64_t backorderCost = 0;
                int64_t holdingCost = 0;
                int64_t orderingCost = 0;
                int64_t unmetDemand = 0;
                int64_t demand = 0;
                int64_t totalDemand = 0;
                int64_t inventoryPosition = 0;
                std::vector<int64_t> inventoryLevelPrior(nrProducts, 0);
                std::vector<int64_t> backorderPrior(nrProducts, 0);
                std::vector<int64_t> backorderPost(nrProducts, 0);
                std::vector<int64_t> orderQtyPrior(nrProducts, 0);
                std::vector<float> itemOrders(nrProducts, 0);
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

                            // create list of number of orders each item is included within
                            if (product.orderQty[leadTime - 1] != 0) {
                                itemOrders[j] += 1;
                            }
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

                            inventoryPosition = product.inventoryLevel;
                            for (int64_t k = 0; k < leadTime; k++) {
                                inventoryPosition += product.orderQty[k];
                            }

                            inventoryPositionsOverTime[j][i] = inventoryPosition;
                            inventoryLevelsOverTime[j][i] = product.inventoryLevel;
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
                itemsPerOrder = std::accumulate(itemOrders.begin(), itemOrders.end(), 0) / periodicity;
                for (int64_t k = 0;k < nrProducts;k++) {
                    itemOrders[k] = itemOrders[k] / periodicity;
                }
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
                kpis.Add("itemsPerOrder", itemsPerOrder);
                kpis.Add("Item1", itemOrders[0]);
                kpis.Add("Item2", itemOrders[1]);
                kpis.Add("Item3", itemOrders[2]);
                kpis.Add("Item4", itemOrders[3]);

                // output evaluation to a file
                auto filename = dp.System().filepath("joint_replenishment", "Evaluation", evaluatePolicyList[k] + "_evaluation.json");
                kpis.SaveToFile(filename);

                // export containers for graphing
                exportToCSV(inventoryPositionsOverTime, "../../GraphData/inventory_positions_" + evaluatePolicyList[k] + ".csv");
                exportToCSV(inventoryLevelsOverTime, "../../GraphData/inventory_levels_" + evaluatePolicyList[k] + ".csv");
            }

        }

        if (dclEvaluate == true) {

            // get trained DCL policy
            std::string gen = "2"; // change to correct gen
            std::cout << mdp->Identifier() << std::endl;
            auto filename = dp.System().filepath(mdp->Identifier(), "dcl_policy_gen" + gen); // DynaPlex_IO folder
            auto path = dp.System().filepath(filename);
            auto dclPolicy = dp.LoadPolicy(mdp, path);

            // initiate state
            mdp->InitiateState({ &trajectory,1 });

            // import data from mdp setup
            int64_t containerVolume;
            int64_t penaltyCost;
            int64_t nrProducts;
            int64_t leadTime;

            mdp_vars_from_json.Get("capacity", containerVolume);
            mdp_vars_from_json.Get("penaltyCost", penaltyCost);
            mdp_vars_from_json.Get("nrProducts", nrProducts);
            mdp_vars_from_json.Get("leadTime", leadTime);

            // containers for graphing
            std::vector<std::vector<int64_t>> inventoryPositionsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));
            std::vector<std::vector<int64_t>> inventoryLevelsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));

            // init tracking KPIs
            float serviceLevel = 0;
            float fillRate = 0;
            float periodicity = 0;
            float itemsPerOrder = 0;
            int64_t backorderCost = 0;
            int64_t holdingCost = 0;
            int64_t orderingCost = 0;
            int64_t unmetDemand = 0;
            int64_t demand = 0;
            int64_t totalDemand = 0;
            int64_t inventoryPosition = 0;
            std::vector<int64_t> inventoryLevelPrior(nrProducts, 0);
            std::vector<int64_t> backorderPrior(nrProducts, 0);
            std::vector<int64_t> backorderPost(nrProducts, 0);
            std::vector<int64_t> orderQtyPrior(nrProducts, 0);
            std::vector<float> itemOrders(nrProducts, 0);
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

                        // create list of number of orders each item is included within
                        if (product.orderQty[leadTime - 1] != 0) {
                            itemOrders[j] += 1;
                        }
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

                        inventoryPosition = product.inventoryLevel;
                        for (int64_t k = 0; k < leadTime; k++) {
                            inventoryPosition += product.orderQty[k];
                        }

                        inventoryPositionsOverTime[j][i] = inventoryPosition;
                        inventoryLevelsOverTime[j][i] = product.inventoryLevel;
                    }
                    i++; // count number of sample periods
                }

                else if (cat.IsAwaitAction()) {
                    dclPolicy->SetAction({ &trajectory,1 });
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
            itemsPerOrder = std::accumulate(itemOrders.begin(), itemOrders.end(), 0) / periodicity;
            for (int64_t k = 0;k < nrProducts;k++) {
                itemOrders[k] = itemOrders[k] / periodicity;
            }
            periodicity = periodicity / sampleNumber;
            serviceLevel = (static_cast<float>(totalDemand) - unmetDemand) / totalDemand;

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
            kpis.Add("itemsPerOrder", itemsPerOrder);
            kpis.Add("Item1", itemOrders[0]);
            kpis.Add("Item2", itemOrders[1]);
            kpis.Add("Item3", itemOrders[2]);
            kpis.Add("Item4", itemOrders[3]);

            // output evaluation to a file
            filename = dp.System().filepath("joint_replenishment", "Evaluation", "dcl_evaluation.json");
            kpis.SaveToFile(filename);

            // export containers for graphing
            exportToCSV(inventoryPositionsOverTime, "../../GraphData/inventory_positions_dcl.csv");
            exportToCSV(inventoryLevelsOverTime, "../../GraphData/inventory_levels_dcl.csv");
        }

        if (ppoEvaluate == true) {

            // get trained ppo policy
            // std::string gen = "2"; // change to correct gen
            auto filename = dp.System().filepath(mdp->Identifier(), "ppo_policy"); // DynaPlex_IO folder
            auto path = dp.System().filepath(filename);
            auto ppoPolicy = dp.LoadPolicy(mdp, path);

            // initiate state
            mdp->InitiateState({ &trajectory,1 });

            // import data from mdp setup
            int64_t containerVolume;
            int64_t penaltyCost;
            int64_t nrProducts;
            int64_t leadTime;

            mdp_vars_from_json.Get("capacity", containerVolume);
            mdp_vars_from_json.Get("penaltyCost", penaltyCost);
            mdp_vars_from_json.Get("nrProducts", nrProducts);
            mdp_vars_from_json.Get("leadTime", leadTime);

            // containers for graphing
            std::vector<std::vector<int64_t>> inventoryPositionsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));
            std::vector<std::vector<int64_t>> inventoryLevelsOverTime(nrProducts, std::vector<int64_t>(sampleNumber));

            // init tracking KPIs
            float serviceLevel = 0;
            float fillRate = 0;
            float periodicity = 0;
            float itemsPerOrder = 0;
            int64_t backorderCost = 0;
            int64_t holdingCost = 0;
            int64_t orderingCost = 0;
            int64_t unmetDemand = 0;
            int64_t demand = 0;
            int64_t totalDemand = 0;
            int64_t inventoryPosition = 0;
            std::vector<int64_t> inventoryLevelPrior(nrProducts, 0);
            std::vector<int64_t> backorderPrior(nrProducts, 0);
            std::vector<int64_t> backorderPost(nrProducts, 0);
            std::vector<int64_t> orderQtyPrior(nrProducts, 0);
            std::vector<float> itemOrders(nrProducts, 0);
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

                        // create list of number of orders each item is included within
                        if (product.orderQty[leadTime - 1] != 0) {
                            itemOrders[j] += 1;
                        }
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

                        inventoryPosition = product.inventoryLevel;
                        for (int64_t k = 0; k < leadTime; k++) {
                            inventoryPosition += product.orderQty[k];
                        }

                        inventoryPositionsOverTime[j][i] = inventoryPosition;
                        inventoryLevelsOverTime[j][i] = product.inventoryLevel;
                    }
                    i++; // count number of sample periods
                }

                else if (cat.IsAwaitAction()) {
                    ppoPolicy->SetAction({ &trajectory,1 });
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
            itemsPerOrder = std::accumulate(itemOrders.begin(), itemOrders.end(), 0) / periodicity;
            for (int64_t k = 0;k < nrProducts;k++) {
                itemOrders[k] = itemOrders[k] / periodicity;
            }
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
            kpis.Add("itemsPerOrder", itemsPerOrder);
            kpis.Add("Item1", itemOrders[0]);
            kpis.Add("Item2", itemOrders[1]);
            kpis.Add("Item3", itemOrders[2]);
            kpis.Add("Item4", itemOrders[3]);

            // output evaluation to a file
            filename = dp.System().filepath("joint_replenishment", "Evaluation", "ppo_evaluation.json");
            kpis.SaveToFile(filename);

            // export containers for graphing
            exportToCSV(inventoryPositionsOverTime, "../../GraphData/inventory_positions_ppo.csv");
            exportToCSV(inventoryLevelsOverTime, "../../GraphData/inventory_levels_ppo.csv");
        }
    }

    catch (const DynaPlex::Error& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

int heatMap() {
    try {

        // get model information
        auto& dp = DynaPlexProvider::Get();
        std::string model_name = "joint_replenishment";
        std::string mdp_config_name = "mdp_config_0.json";

        std::string file_path = dp.System().filepath("mdp_config_examples", model_name, mdp_config_name);
        auto mdp_vars_from_json = VarGroup::LoadFromFile(file_path);
        auto mdp = dp.GetMDP(mdp_vars_from_json);

        auto policy = mdp->GetPolicy("random");
        
        // change one to be true; set the others to false
        bool ruleBasedEvaluate = false;
        bool dclEvaluate = true;
        bool ppoEvaluate = false;

        if (ruleBasedEvaluate == true) {
            // get rule-based policy
            std::vector<std::string> evaluatePolicyList = { "canOrderPolicy" };
            std::string policy_config_name = "policy_config_0.json"; // 1 = can order 2 = periodic review
            std::string policy_file_path = dp.System().filepath("mdp_config_examples", model_name, policy_config_name);
            VarGroup evaluatePolicy = VarGroup::LoadFromFile(policy_file_path);
            auto policy = mdp->GetPolicy(evaluatePolicy);
        }

        if (dclEvaluate == true) {
            // get trained DCL policy
            std::string gen = "10"; // change to correct gen
            std::cout << mdp->Identifier() << std::endl;
            auto filename = dp.System().filepath(mdp->Identifier(), "dcl_policy_gen" + gen); // DynaPlex_IO folder
            auto path = dp.System().filepath(filename);
            policy = dp.LoadPolicy(mdp, path); // udpate policy from random
        }

        if (ppoEvaluate == true) {
            // get ppo policy
            auto filename = dp.System().filepath(mdp->Identifier(), "ppo_policy"); // DynaPlex_IO folder
            auto path = dp.System().filepath(filename);
            auto ppoPolicy = dp.LoadPolicy(mdp, path);
        }

        //get some initializing variables from json
        std::vector<double> initialForecast;
        std::vector<double> initialSigma;
        int64_t nrProducts;
        int64_t leadTime;

        mdp_vars_from_json.Get("initialForecast", initialForecast);
        mdp_vars_from_json.Get("initialSigma", initialSigma);
        mdp_vars_from_json.Get("nrProducts", nrProducts);
        mdp_vars_from_json.Get("leadTime", leadTime);

        // define heat map parameters
        int64_t item1 = 0;
        int64_t item2 = 3;
        int64_t maxInventoryPosition = 90; // set by looking at inventory plot
        int64_t actionIndex = 0; // 0 = order; 1 = amount
        int64_t inventory2 = 25;
        int64_t inventory3 = 45;
        int64_t inventory4 = 15;
        int64_t order1;
        int64_t order2;
        int64_t order3;

        // define container variables
        std::vector<int64_t> actions;

        for (int inventoryPosition1 = -10; inventoryPosition1 < maxInventoryPosition; inventoryPosition1++) {
            for (int inventoryPosition2 = -10; inventoryPosition2 < maxInventoryPosition; inventoryPosition2++) {
                
                DynaPlex::VarGroup stateVars{
                    {"cat", StateCategory::AwaitAction(0).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                    {"SKUs", std::vector<DynaPlex::VarGroup>{
                        DynaPlex::VarGroup{
                            {"skuNumber", 0},
                            {"forecastedDemand", initialForecast[0]},
                            {"forecastDeviation", initialSigma[0]},
                            {"inventoryLevel", inventoryPosition1},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        },
                        DynaPlex::VarGroup{
                            {"skuNumber", 1},
                            {"forecastedDemand", initialForecast[1]},
                            {"forecastDeviation", initialSigma[1]},
                            {"inventoryLevel", inventoryPosition2},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        },
                        DynaPlex::VarGroup{
                            {"skuNumber", 2},
                            {"forecastedDemand", initialForecast[2]},
                            {"forecastDeviation", initialSigma[2]},
                            {"inventoryLevel", inventory3},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        },
                        DynaPlex::VarGroup{
                            {"skuNumber", 3},
                            {"forecastedDemand", initialForecast[3]},
                            {"forecastDeviation", initialSigma[3]},
                            {"inventoryLevel", inventory4},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        }
                    }},
                    {"usedCapacity", 0},
                    {"remainingEvents", 100},
                    {"orderItem", item1},
                    {"periodOrderingCosts", 0},
                    {"periodHoldingCosts", 0},
                    {"periodBackorderCosts", 0},
                    {"periodCount", 1000},
                };

                auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                std::vector<Trajectory> trajVec(1);
                mdp->InitiateState({ &trajVec[0] ,1 }, state);
                policy->SetAction(trajVec);

                // pass if no order will be made
                if (trajVec[0].NextAction == 0) {
                    actions.push_back(trajVec[0].NextAction);
                }
                
                // first simulate item 1 being ordered
                else {
                    DynaPlex::VarGroup stateVars{
                    {"cat", StateCategory::AwaitAction(1).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                    {"SKUs", std::vector<DynaPlex::VarGroup>{
                        DynaPlex::VarGroup{
                            {"skuNumber", 0},
                            {"forecastedDemand", initialForecast[0]},
                            {"forecastDeviation", initialSigma[0]},
                            {"inventoryLevel", inventoryPosition1},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        },
                        DynaPlex::VarGroup{
                            {"skuNumber", 1},
                            {"forecastedDemand", initialForecast[1]},
                            {"forecastDeviation", initialSigma[1]},
                            {"inventoryLevel", inventoryPosition2},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        },
                        DynaPlex::VarGroup{
                            {"skuNumber", 2},
                            {"forecastedDemand", initialForecast[2]},
                            {"forecastDeviation", initialSigma[2]},
                            {"inventoryLevel", inventory3},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        },
                        DynaPlex::VarGroup{
                            {"skuNumber", 3},
                            {"forecastedDemand", initialForecast[3]},
                            {"forecastDeviation", initialSigma[3]},
                            {"inventoryLevel", inventory4},
                            {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                        }
                    }},
                    {"usedCapacity", 0},
                    {"remainingEvents", 100},
                    {"orderItem", item1},
                    {"periodOrderingCosts", 0},
                    {"periodHoldingCosts", 0},
                    {"periodBackorderCosts", 0},
                    {"periodCount", 1000},
                    };

                    auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                    std::vector<Trajectory> trajVec(1);
                    mdp->InitiateState({ &trajVec[0] ,1 }, state);
                    policy->SetAction(trajVec);
                    order1 = trajVec[0].NextAction;


                    // simulate how the ordering process works
                    if (item2 >= 1) {
                        DynaPlex::VarGroup stateVars{
                        {"cat", StateCategory::AwaitAction(0).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                        {"SKUs", std::vector<DynaPlex::VarGroup>{
                            DynaPlex::VarGroup{
                                {"skuNumber", 0},
                                {"forecastedDemand", initialForecast[0]},
                                {"forecastDeviation", initialSigma[0]},
                                {"inventoryLevel", inventoryPosition1},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order1}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 1},
                                {"forecastedDemand", initialForecast[1]},
                                {"forecastDeviation", initialSigma[1]},
                                {"inventoryLevel", inventoryPosition2},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 2},
                                {"forecastedDemand", initialForecast[2]},
                                {"forecastDeviation", initialSigma[2]},
                                {"inventoryLevel", inventory3},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 3},
                                {"forecastedDemand", initialForecast[3]},
                                {"forecastDeviation", initialSigma[3]},
                                {"inventoryLevel", inventory4},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            }
                        }},
                        {"usedCapacity", 0},
                        {"remainingEvents", 100},
                        {"orderItem", item1},
                        {"periodOrderingCosts", 0},
                        {"periodHoldingCosts", 0},
                        {"periodBackorderCosts", 0},
                        {"periodCount", 1000},
                        };

                        auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                        std::vector<Trajectory> trajVec(1);
                        mdp->InitiateState({ &trajVec[0] ,1 }, state);
                        policy->SetAction(trajVec);
                    }

                    if (item2 >= 1 && actionIndex == 1 && trajVec[0].NextAction == 1) { // if action = 1; item 2 should be ordered
                        DynaPlex::VarGroup stateVars{
                        {"cat", StateCategory::AwaitAction(1).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                        {"SKUs", std::vector<DynaPlex::VarGroup>{
                            DynaPlex::VarGroup{
                                {"skuNumber", 0},
                                {"forecastedDemand", initialForecast[0]},
                                {"forecastDeviation", initialSigma[0]},
                                {"inventoryLevel", inventoryPosition1},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order1}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 1},
                                {"forecastedDemand", initialForecast[1]},
                                {"forecastDeviation", initialSigma[1]},
                                {"inventoryLevel", inventoryPosition2},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 2},
                                {"forecastedDemand", initialForecast[2]},
                                {"forecastDeviation", initialSigma[2]},
                                {"inventoryLevel", inventory3},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 3},
                                {"forecastedDemand", initialForecast[3]},
                                {"forecastDeviation", initialSigma[3]},
                                {"inventoryLevel", inventory4},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            }
                        }},
                        {"usedCapacity", 0},
                        {"remainingEvents", 100},
                        {"orderItem", 1},
                        {"periodOrderingCosts", 0},
                        {"periodHoldingCosts", 0},
                        {"periodBackorderCosts", 0},
                        {"periodCount", 1000},
                        };

                        auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                        std::vector<Trajectory> trajVec(1);
                        mdp->InitiateState({ &trajVec[0] ,1 }, state);
                        policy->SetAction(trajVec);
                        order2 = trajVec[0].NextAction;
                    }

                    if (item2 >= 2) {
                        DynaPlex::VarGroup stateVars{
                        {"cat", StateCategory::AwaitAction(0).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                        {"SKUs", std::vector<DynaPlex::VarGroup>{
                            DynaPlex::VarGroup{
                                {"skuNumber", 0},
                                {"forecastedDemand", initialForecast[0]},
                                {"forecastDeviation", initialSigma[0]},
                                {"inventoryLevel", inventoryPosition1},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order1}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 1},
                                {"forecastedDemand", initialForecast[1]},
                                {"forecastDeviation", initialSigma[1]},
                                {"inventoryLevel", inventoryPosition2},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order2}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 2},
                                {"forecastedDemand", initialForecast[2]},
                                {"forecastDeviation", initialSigma[2]},
                                {"inventoryLevel", inventory3},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 3},
                                {"forecastedDemand", initialForecast[3]},
                                {"forecastDeviation", initialSigma[3]},
                                {"inventoryLevel", inventory4},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            }
                        }},
                        {"usedCapacity", 0},
                        {"remainingEvents", 100},
                        {"orderItem", item1},
                        {"periodOrderingCosts", 0},
                        {"periodHoldingCosts", 0},
                        {"periodBackorderCosts", 0},
                        {"periodCount", 1000},
                        };

                        auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                        std::vector<Trajectory> trajVec(1);
                        mdp->InitiateState({ &trajVec[0] ,1 }, state);
                        policy->SetAction(trajVec);
                    }

                    if (item2 >= 2 && actionIndex == 1 && trajVec[0].NextAction == 2) { // if action = 2; item 3 should be ordered
                        DynaPlex::VarGroup stateVars{
                        {"cat", StateCategory::AwaitAction(1).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                        {"SKUs", std::vector<DynaPlex::VarGroup>{
                            DynaPlex::VarGroup{
                                {"skuNumber", 0},
                                {"forecastedDemand", initialForecast[0]},
                                {"forecastDeviation", initialSigma[0]},
                                {"inventoryLevel", inventoryPosition1},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order1}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 1},
                                {"forecastedDemand", initialForecast[1]},
                                {"forecastDeviation", initialSigma[1]},
                                {"inventoryLevel", inventoryPosition2},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order2}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 2},
                                {"forecastedDemand", initialForecast[2]},
                                {"forecastDeviation", initialSigma[2]},
                                {"inventoryLevel", inventory3},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 3},
                                {"forecastedDemand", initialForecast[3]},
                                {"forecastDeviation", initialSigma[3]},
                                {"inventoryLevel", inventory4},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            }
                        }},
                        {"usedCapacity", 0},
                        {"remainingEvents", 100},
                        {"orderItem", 2},
                        {"periodOrderingCosts", 0},
                        {"periodHoldingCosts", 0},
                        {"periodBackorderCosts", 0},
                        {"periodCount", 1000},
                        };

                        auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                        std::vector<Trajectory> trajVec(1);
                        mdp->InitiateState({ &trajVec[0] ,1 }, state);
                        policy->SetAction(trajVec);
                        order3 = trajVec[0].NextAction;
                    }

                    if (item2 >= 3) {
                        DynaPlex::VarGroup stateVars{
                        {"cat", StateCategory::AwaitAction(0).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                        {"SKUs", std::vector<DynaPlex::VarGroup>{
                            DynaPlex::VarGroup{
                                {"skuNumber", 0},
                                {"forecastedDemand", initialForecast[0]},
                                {"forecastDeviation", initialSigma[0]},
                                {"inventoryLevel", inventoryPosition1},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order1}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 1},
                                {"forecastedDemand", initialForecast[1]},
                                {"forecastDeviation", initialSigma[1]},
                                {"inventoryLevel", inventoryPosition2},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order2}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 2},
                                {"forecastedDemand", initialForecast[2]},
                                {"forecastDeviation", initialSigma[2]},
                                {"inventoryLevel", inventory3},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order3}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 3},
                                {"forecastedDemand", initialForecast[3]},
                                {"forecastDeviation", initialSigma[3]},
                                {"inventoryLevel", inventory4},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            }
                        }},
                        {"usedCapacity", 0},
                        {"remainingEvents", 100},
                        {"orderItem", 2},
                        {"periodOrderingCosts", 0},
                        {"periodHoldingCosts", 0},
                        {"periodBackorderCosts", 0},
                        {"periodCount", 1000},
                        };

                        auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                        std::vector<Trajectory> trajVec(1);
                        mdp->InitiateState({ &trajVec[0] ,1 }, state);
                        policy->SetAction(trajVec);
                    }

                    if (item2 >= 3 && actionIndex == 1 && trajVec[0].NextAction == 3) { // if action = 3; item 4 should be ordered
                        DynaPlex::VarGroup stateVars{
                        {"cat", StateCategory::AwaitAction(1).ToVarGroup()}, // AwaitAction(0) -> item selection AwaitAction(1) -> order quantity
                        {"SKUs", std::vector<DynaPlex::VarGroup>{
                            DynaPlex::VarGroup{
                                {"skuNumber", 0},
                                {"forecastedDemand", initialForecast[0]},
                                {"forecastDeviation", initialSigma[0]},
                                {"inventoryLevel", inventoryPosition1},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order1}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 1},
                                {"forecastedDemand", initialForecast[1]},
                                {"forecastDeviation", initialSigma[1]},
                                {"inventoryLevel", inventoryPosition2},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order2}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 2},
                                {"forecastedDemand", initialForecast[2]},
                                {"forecastDeviation", initialSigma[2]},
                                {"inventoryLevel", inventory3},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, order3}}
                            },
                            DynaPlex::VarGroup{
                                {"skuNumber", 3},
                                {"forecastedDemand", initialForecast[3]},
                                {"forecastDeviation", initialSigma[3]},
                                {"inventoryLevel", inventory4},
                                {"orderQty", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
                            }
                        }},
                        {"usedCapacity", 0},
                        {"remainingEvents", 100},
                        {"orderItem", 3},
                        {"periodOrderingCosts", 0},
                        {"periodHoldingCosts", 0},
                        {"periodBackorderCosts", 0},
                        {"periodCount", 1000},
                        };

                        auto state = mdp->GetState(stateVars); // use stateVars input to generate state
                        std::vector<Trajectory> trajVec(1);
                        mdp->InitiateState({ &trajVec[0] ,1 }, state);
                        policy->SetAction(trajVec);
                    }

                    actions.push_back(trajVec[0].NextAction);
                }
            };

            VarGroup kpis{};
            kpis.Add("actions", actions);

            auto filename = dp.System().filepath("joint_replenishment", "Evaluation", "heatmap_items" + std::to_string(item1) + std::to_string(item2) + ".json");
            kpis.SaveToFile(filename);
        }
    }

    catch (const DynaPlex::Error& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

int main() {

    // train();
    evaluate();
    // heatMap();
};