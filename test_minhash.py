import unittest
import pytest
from minhash import MinHash

class minHashTest(unittest.TestCase):
    def test_method1(self):
        mhs = MinHash(4, cache_size=10)
        
        corpus = [
            "pytest features in unittest.TestCase subclasses",
            "Pytest Features in Unittest.TestCase Subclasses",
            "How to Use Unittest-Based Tests With Pytest",
            "How to use unittest-based tests with pytest",
            "The matrix product of the inputs. This is a scalar only when both x1, x2 are 1-d vectors.",
            "Although it’s usually better to explicitly declare use of fixtures you need for a given test, you may sometimes want to have fixtures that are automatically used in a given context.",
            "You can also gradually move away from subclassing from unittest.TestCase to plain asserts and then start to benefit from the full pytest feature set step by step.",
            "The matmul function implements the semantics of the @ operator introduced in Python 3.5 following PEP 465."
        ]
        
        pairs = mhs.generate_score_pairs(corpus)
        
        assert len(pairs)==10
        assert pairs[0]==(2,3,1.0)
        assert pairs[1]==(0,1,1.0)
        assert pairs[2]==(1,5,0.25)
        assert pairs[3]==(0,5,0.25)
        
        # print(pairs)
