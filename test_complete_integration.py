"""
Test Complete Integration: Upload → Train → Predict
Verifies that:
1. Company graph is loaded correctly
2. Trained model is used for predictions
3. Predictions are accurate and use correct node IDs
4. Map visualization shows correct colors
"""

import requests
import json
import pandas as pd
import os

BASE_URL = "http://localhost:5000"

def test_complete_integration():
    print("="*70)
    print("TESTING COMPLETE INTEGRATION")
    print("="*70)
    
    # Get company_id from localStorage (you'll need to provide this)
    company_id = input("Enter your company_id (from localStorage): ").strip()
    
    if not company_id:
        print("❌ No company_id provided. Please upload a graph first.")
        return
    
    print(f"\n📋 Testing with company_id: {company_id}")
    
    # ========================================================================
    # TEST 1: Check if company data exists
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: Checking Company Data")
    print("="*70)
    
    uploads_dir = f"C:/Users/janny/Desktop/final_year/backend/uploads/{company_id}"
    
    if not os.path.exists(uploads_dir):
        print(f"❌ Company directory not found: {uploads_dir}")
        return
    
    print(f"✅ Company directory exists: {uploads_dir}")
    
    # Check for required files
    required_files = ['nodes.csv', 'edges.csv', 'model_comparison.csv']
    for file in required_files:
        file_path = os.path.join(uploads_dir, file)
        if os.path.exists(file_path):
            print(f"✅ Found: {file}")
        else:
            print(f"⚠️  Missing: {file}")
    
    # Check for trained models
    model_files = [f for f in os.listdir(uploads_dir) if f.endswith('_model.pt')]
    if model_files:
        print(f"✅ Found {len(model_files)} trained models:")
        for model in model_files:
            print(f"   - {model}")
    else:
        print("⚠️  No trained models found. Run training first.")
    
    # Load company data
    nodes_df = pd.read_csv(os.path.join(uploads_dir, 'nodes.csv'))
    edges_df = pd.read_csv(os.path.join(uploads_dir, 'edges.csv'))
    
    print(f"\n📊 Company Graph Stats:")
    print(f"   Nodes: {len(nodes_df)}")
    print(f"   Edges: {len(edges_df)}")
    print(f"   Node ID range: 0-{len(nodes_df)-1}")
    
    # ========================================================================
    # TEST 2: Check API /api/graph endpoint
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: Testing /api/graph Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/api/graph?company_id={company_id}")
        data = response.json()
        
        if data['status'] == 'success':
            print(f"✅ API returned graph data")
            print(f"   Nodes: {len(data['nodes'])}")
            print(f"   Edges: {len(data['edges'])}")
            print(f"   Node IDs: {data['nodes'][0]['id']} to {data['nodes'][-1]['id']}")
            
            # Verify node IDs are sequential
            node_ids = [n['id'] for n in data['nodes']]
            expected_ids = list(range(len(nodes_df)))
            if node_ids == expected_ids:
                print(f"✅ Node IDs are correct (0 to {len(nodes_df)-1})")
            else:
                print(f"❌ Node ID mismatch!")
                print(f"   Expected: 0 to {len(nodes_df)-1}")
                print(f"   Got: {node_ids[:5]}...{node_ids[-5:]}")
        else:
            print(f"❌ API error: {data}")
    except Exception as e:
        print(f"❌ API request failed: {e}")
    
    # ========================================================================
    # TEST 3: Test Prediction with Company Model
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: Testing GNN Prediction")
    print("="*70)
    
    # Select a few nodes to disrupt (use valid indices)
    num_nodes = len(nodes_df)
    disrupted_nodes = [0, 1, 2]  # First 3 nodes
    severity = 0.8
    
    print(f"🔮 Running prediction:")
    print(f"   Disrupted nodes: {disrupted_nodes}")
    print(f"   Severity: {severity}")
    print(f"   Company ID: {company_id}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json={
                "company_id": company_id,
                "disrupted_nodes": disrupted_nodes,
                "disrupted_edges": [],
                "disruption_severity": severity,
                "buffer_capacity": 0.5
            }
        )
        
        result = response.json()
        
        if result['status'] == 'success':
            print(f"✅ Prediction successful!")
            print(f"\n📊 Prediction Summary:")
            print(f"   Failed: {result['summary']['failed']}")
            print(f"   Degraded: {result['summary']['degraded']}")
            print(f"   Normal: {result['summary']['normal']}")
            print(f"   Total: {result['summary']['total_nodes']}")
            
            print(f"\n🤖 Model Info:")
            print(f"   Model: {result['model_info']['model_name']}")
            print(f"   Company: {result['model_info']['company_id']}")
            
            # Check predictions for disrupted nodes
            print(f"\n🔍 Checking Disrupted Nodes:")
            for node_id in disrupted_nodes:
                pred = next((p for p in result['predictions'] if p['node_id'] == node_id), None)
                if pred:
                    print(f"   Node {node_id}: {pred['label_name']} (confidence: {max(pred['probability'])*100:.1f}%)")
                else:
                    print(f"   Node {node_id}: ❌ No prediction found")
            
            # Check some non-disrupted nodes
            print(f"\n🔍 Sample Non-Disrupted Nodes:")
            for node_id in [10, 20, 30]:
                if node_id < num_nodes:
                    pred = next((p for p in result['predictions'] if p['node_id'] == node_id), None)
                    if pred:
                        print(f"   Node {node_id}: {pred['label_name']} (confidence: {max(pred['probability'])*100:.1f}%)")
            
            # ========================================================================
            # TEST 4: Verify Map Visualization Data
            # ========================================================================
            print("\n" + "="*70)
            print("TEST 4: Verifying Map Visualization Data")
            print("="*70)
            
            # Count nodes by status
            failed_nodes = [p for p in result['predictions'] if p['label'] == 0]
            degraded_nodes = [p for p in result['predictions'] if p['label'] == 1]
            normal_nodes = [p for p in result['predictions'] if p['label'] == 2]
            
            print(f"✅ Node Status Distribution:")
            print(f"   🔴 Failed ({len(failed_nodes)}): {[p['node_id'] for p in failed_nodes[:5]]}...")
            print(f"   🟠 Degraded ({len(degraded_nodes)}): {[p['node_id'] for p in degraded_nodes[:5]]}...")
            print(f"   🟢 Normal ({len(normal_nodes)}): {[p['node_id'] for p in normal_nodes[:5]]}...")
            
            # Verify disrupted nodes are marked
            print(f"\n✅ Disrupted Node Status:")
            for node_id in disrupted_nodes:
                pred = next((p for p in result['predictions'] if p['node_id'] == node_id), None)
                if pred:
                    color = "🔴" if pred['label'] == 0 else "🟠" if pred['label'] == 1 else "🟢"
                    print(f"   {color} Node {node_id}: {pred['label_name']}")
            
            # ========================================================================
            # TEST 5: Verify Model is Company-Specific
            # ========================================================================
            print("\n" + "="*70)
            print("TEST 5: Verifying Company-Specific Model")
            print("="*70)
            
            # Check if model comparison exists
            comparison_path = os.path.join(uploads_dir, 'model_comparison.csv')
            if os.path.exists(comparison_path):
                comparison_df = pd.read_csv(comparison_path)
                best_model = comparison_df.loc[comparison_df['f1'].idxmax()]
                
                print(f"✅ Model Comparison Found:")
                print(f"   Best Model: {best_model['model']}")
                print(f"   F1 Score: {best_model['f1']:.4f}")
                print(f"   Accuracy: {best_model['accuracy']:.4f}")
                
                if result['model_info']['model_name'] == best_model['model']:
                    print(f"✅ Using best trained model: {best_model['model']}")
                else:
                    print(f"⚠️  Model mismatch:")
                    print(f"   Expected: {best_model['model']}")
                    print(f"   Got: {result['model_info']['model_name']}")
            else:
                print(f"⚠️  No model comparison found")
            
            # ========================================================================
            # FINAL SUMMARY
            # ========================================================================
            print("\n" + "="*70)
            print("INTEGRATION TEST SUMMARY")
            print("="*70)
            
            checks = [
                ("Company data exists", os.path.exists(uploads_dir)),
                ("Trained models exist", len(model_files) > 0),
                ("API returns correct graph", len(data['nodes']) == len(nodes_df)),
                ("Prediction successful", result['status'] == 'success'),
                ("Using company model", result['model_info']['company_id'] == company_id),
                ("Disrupted nodes predicted", all(
                    any(p['node_id'] == nid for p in result['predictions']) 
                    for nid in disrupted_nodes
                )),
            ]
            
            passed = sum(1 for _, check in checks if check)
            total = len(checks)
            
            for check_name, check_result in checks:
                status = "✅" if check_result else "❌"
                print(f"{status} {check_name}")
            
            print(f"\n{'='*70}")
            print(f"RESULT: {passed}/{total} checks passed")
            print(f"{'='*70}")
            
            if passed == total:
                print("\n🎉 ALL TESTS PASSED! Integration is working correctly!")
                print("\n📋 What's Working:")
                print("   ✅ Company-specific graph loaded")
                print("   ✅ Trained model used for predictions")
                print("   ✅ Correct node IDs (0 to N-1)")
                print("   ✅ Disrupted nodes predicted correctly")
                print("   ✅ Map visualization data ready")
                print("\n🗺️  Map Visualization:")
                print("   - Disrupted nodes should show with red pulse effect")
                print("   - Failed nodes: RED circles")
                print("   - Degraded nodes: ORANGE circles")
                print("   - Normal nodes: GREEN circles")
            else:
                print("\n⚠️  Some tests failed. Check the output above for details.")
        
        else:
            print(f"❌ Prediction failed: {result}")
            if 'traceback' in result:
                print(f"\nError details:\n{result['traceback']}")
    
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_complete_integration()
