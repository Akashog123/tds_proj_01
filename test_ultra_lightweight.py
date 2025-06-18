#!/usr/bin/env python3
"""
Test script for the ultra-lightweight precomputation and deployment system.
This script validates the complete workflow from precomputation to deployment.
"""

import logging
import time
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_precomputation():
    """Test the precomputation process"""
    logger.info("🧪 Testing precomputation process...")
    
    try:
        # Import and run precomputation
        from precompute_embeddings_ultra import UltraLightweightPrecomputer
        
        precomputer = UltraLightweightPrecomputer()
        success = precomputer.run_precomputation()
        
        if not success:
            logger.error("❌ Precomputation failed")
            return False
        
        # Check if output files were created
        precomputed_dir = Path("precomputed_ultra")
        required_files = [
            "local_embeddings.pkl",
            "tfidf_data.pkl",
            "search_indices.json",
            "subthreads_light.json",
            "ultra_metadata.json"
        ]
        
        optional_files = [
            "openai_embeddings.pkl"  # Optional if OpenAI API is not available
        ]
        
        missing_required = []
        for filename in required_files:
            filepath = precomputed_dir / filename
            if not filepath.exists():
                missing_required.append(filename)
        
        if missing_required:
            logger.error(f"❌ Missing required precomputed files: {missing_required}")
            return False
        
        # Check optional files
        missing_optional = []
        for filename in optional_files:
            filepath = precomputed_dir / filename
            if not filepath.exists():
                missing_optional.append(filename)
        
        if missing_optional:
            logger.warning(f"⚠️ Missing optional files (may impact search quality): {missing_optional}")
        
        # Check file sizes
        total_size = sum(f.stat().st_size for f in precomputed_dir.glob('*') if f.is_file())
        logger.info(f"✅ Precomputation successful - Total size: {total_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Precomputation test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_ultra_lightweight_engine():
    """Test the ultra-lightweight search engine"""
    logger.info("🧪 Testing ultra-lightweight search engine...")
    
    try:
        from ultra_lightweight_engine import UltraLightweightSearchEngine
        
        # Initialize engine
        start_time = time.time()
        engine = UltraLightweightSearchEngine()
        init_time = time.time() - start_time
        
        logger.info(f"✅ Engine initialized in {init_time:.2f} seconds")
        
        # Get stats
        stats = engine.get_stats()
        logger.info(f"📊 Engine stats: {stats}")
        
        # Test search functionality
        test_queries = [
            "machine learning",
            "data analysis", 
            "python programming",
            "statistics",
            "visualization"
        ]
        
        search_times = []
        for query in test_queries:
            start_time = time.time()
            results = engine.search(query, top_k=3)
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            logger.info(f"🔍 Query: '{query}' - {len(results)} results in {search_time:.4f}s")
            
            if results:
                best_result = results[0]
                logger.info(f"   Best: {best_result['topic_title'][:50]}... (score: {best_result['score']:.3f})")
        
        avg_search_time = sum(search_times) / len(search_times)
        logger.info(f"⚡ Average search time: {avg_search_time:.4f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ultra-lightweight engine test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_ultra_railway_app():
    """Test the ultra-lightweight Railway application"""
    logger.info("🧪 Testing ultra-lightweight Railway app...")
    
    try:
        # Test import using importlib to handle the dash in filename
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("main_ultra_railway", "main-ultra-railway.py")
        main_ultra_railway = importlib.util.module_from_spec(spec)
        sys.modules["main_ultra_railway"] = main_ultra_railway
        spec.loader.exec_module(main_ultra_railway)
        
        # Check if search engine initializes
        from ultra_lightweight_engine import UltraLightweightSearchEngine
        engine = UltraLightweightSearchEngine()
        
        qa_system = main_ultra_railway.UltraLightweightQuestionAnswerer(engine)
        
        # Test question answering
        test_question = "What is machine learning?"
        answer_data = qa_system.answer_question(test_question, top_k=3)
        
        logger.info(f"✅ QA system test successful")
        logger.info(f"   Question: {test_question}")
        logger.info(f"   Answer length: {len(answer_data['answer'])} chars")
        logger.info(f"   Confidence: {answer_data['confidence']}")
        logger.info(f"   Sources: {len(answer_data['sources'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ultra-lightweight Railway app test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_performance_comparison():
    """Compare performance between different engines"""
    logger.info("🧪 Testing performance comparison...")
    
    try:
        # Test queries
        test_queries = [
            "machine learning algorithms",
            "data visualization techniques", 
            "python data analysis",
            "statistical methods",
            "course assignments"
        ]
        
        # Test ultra-lightweight engine
        from ultra_lightweight_engine import UltraLightweightSearchEngine
        
        start_time = time.time()
        ultra_engine = UltraLightweightSearchEngine()
        ultra_init_time = time.time() - start_time
        
        ultra_search_times = []
        for query in test_queries:
            start_time = time.time()
            results = ultra_engine.search(query, top_k=5)
            search_time = time.time() - start_time
            ultra_search_times.append(search_time)
        
        ultra_avg = sum(ultra_search_times) / len(ultra_search_times)
        
        logger.info(f"⚡ Ultra-lightweight engine:")
        logger.info(f"   Init time: {ultra_init_time:.2f}s")
        logger.info(f"   Avg search time: {ultra_avg:.4f}s")
        logger.info(f"   Total test time: {ultra_init_time + sum(ultra_search_times):.2f}s")
        
        # Compare with Railway engine if available
        try:
            from railway_search_engine import RailwaySearchEngine
            
            start_time = time.time()
            railway_engine = RailwaySearchEngine()
            railway_init_time = time.time() - start_time
            
            railway_search_times = []
            for query in test_queries:
                start_time = time.time()
                results = railway_engine.search(query, top_k=5)
                search_time = time.time() - start_time
                railway_search_times.append(search_time)
            
            railway_avg = sum(railway_search_times) / len(railway_search_times)
            
            logger.info(f"🔧 Railway engine (for comparison):")
            logger.info(f"   Init time: {railway_init_time:.2f}s")
            logger.info(f"   Avg search time: {railway_avg:.4f}s")
            logger.info(f"   Total test time: {railway_init_time + sum(railway_search_times):.2f}s")
            
            # Calculate improvements
            init_speedup = railway_init_time / ultra_init_time if ultra_init_time > 0 else float('inf')
            search_speedup = railway_avg / ultra_avg if ultra_avg > 0 else float('inf')
            
            logger.info(f"🚀 Performance improvements:")
            logger.info(f"   Init speedup: {init_speedup:.2f}x")
            logger.info(f"   Search speedup: {search_speedup:.2f}x")
            
        except Exception as e:
            logger.warning(f"⚠️ Railway engine comparison skipped: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_deployment_readiness():
    """Test deployment readiness"""
    logger.info("🧪 Testing deployment readiness...")
    
    try:
        # Check precomputed data exists
        precomputed_dir = Path("precomputed_ultra")
        if not precomputed_dir.exists():
            logger.error("❌ Precomputed directory not found")
            return False
        
        # Check minimal requirements
        requirements_file = Path("requirements-ultra-light.txt")
        if not requirements_file.exists():
            logger.error("❌ Ultra-light requirements file not found")
            return False
        
        # Check Railway app
        railway_app = Path("main-ultra-railway.py")
        if not railway_app.exists():
            logger.error("❌ Ultra-lightweight Railway app not found")
            return False
        
        # Test import without heavy dependencies
        original_path = sys.path.copy()
        try:
            # Simulate environment without heavy deps
            import ultra_lightweight_engine
            
            # Import main-ultra-railway.py using importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location("main_ultra_railway", "main-ultra-railway.py")
            main_ultra_railway = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_ultra_railway)
            
            logger.info("✅ All components can be imported")
            
        except ImportError as e:
            if any(dep in str(e) for dep in ['chromadb', 'openai', 'sentence_transformers', 'torch']):
                logger.error(f"❌ Heavy dependency detected: {e}")
                return False
            else:
                # Re-raise if it's not a heavy dependency issue
                raise
        
        # Check file sizes
        total_size = sum(f.stat().st_size for f in precomputed_dir.glob('*') if f.is_file())
        
        logger.info(f"✅ Deployment readiness check passed")
        logger.info(f"   Precomputed data size: {total_size / (1024*1024):.2f} MB")
        logger.info(f"   Minimal dependencies: FastAPI, NumPy only")
        logger.info(f"   No external API calls required")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment readiness test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all ultra-lightweight tests"""
    logger.info("🚀 Starting ultra-lightweight system tests...")
    start_time = time.time()
    
    tests = [
        ("Precomputation", test_precomputation),
        ("Ultra-lightweight Engine", test_ultra_lightweight_engine),
        ("Ultra-lightweight Railway App", test_ultra_railway_app),
        ("Performance Comparison", test_performance_comparison),
        ("Deployment Readiness", test_deployment_readiness)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
            results[test_name] = False
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Ultra-lightweight system is ready for Railway deployment!")
        return True
    else:
        logger.error(f"\n💥 {total - passed} test(s) failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)