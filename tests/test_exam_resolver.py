import unittest

from agent.exam_resolver import (
    ALIAS,
    EQUIVALENT,
    PARTIAL_SUBSTITUTE,
    UNRESOLVED,
    ExamResolver,
)


class ExamResolverTests(unittest.TestCase):
    def test_alias_resolution_is_full_coverage(self):
        resolver = ExamResolver(
            catalog_names=["骨髓流式细胞免疫表型分析"],
            aliases={"流式细胞术免疫分型": "骨髓流式细胞免疫表型分析"},
        )

        result = resolver.resolve("流式细胞术免疫分型")

        self.assertEqual(result.resolution_type, ALIAS)
        self.assertEqual(result.resolved_exam, "骨髓流式细胞免疫表型分析")
        self.assertEqual(result.diagnostic_coverage, 1.0)

    def test_partial_substitute_is_not_full_confirmation(self):
        resolver = ExamResolver(catalog_names=["骨髓穿刺和活检（BMAB）", "全血细胞计数（CBC）"])

        result = resolver.resolve("流式细胞术免疫分型")

        self.assertEqual(result.resolution_type, PARTIAL_SUBSTITUTE)
        self.assertEqual(result.resolved_exam, "骨髓穿刺和活检（BMAB）")
        self.assertLess(result.diagnostic_coverage, 1.0)
        self.assertNotEqual(result.resolved_exam, "全血细胞计数（CBC）")

    def test_special_pulmonary_vascular_request_resolves_as_partial_if_only_ct_exists(self):
        resolver = ExamResolver(catalog_names=["胸部CT扫描（Chest CT）"])

        result = resolver.resolve("肺动脉CTA")

        self.assertEqual(result.resolution_type, PARTIAL_SUBSTITUTE)
        self.assertEqual(result.resolved_exam, "胸部CT扫描（Chest CT）")
        self.assertLess(result.diagnostic_coverage, 1.0)

    def test_pavm_candidate_preserves_controlled_cta_request(self):
        resolver = ExamResolver(
            catalog_names=["\u80f8\u90e8CT\u626b\u63cf\uff08Chest CT\uff09"]
        )

        result = resolver.resolve(
            "\u80ba\u52a8\u8109CTA",
            candidate="\u80ba\u52a8\u9759\u8109\u7618",
        )

        self.assertEqual(result.resolution_type, EQUIVALENT)
        self.assertEqual(result.resolved_exam, "\u80ba\u52a8\u8109CTA")
        self.assertEqual(result.diagnostic_coverage, 1.0)

    def test_leukemia_candidate_preserves_controlled_marrow_and_flow_requests(self):
        resolver = ExamResolver(
            catalog_names=[
                "\u7ec4\u7ec7\u75c5\u7406\u5b66\u68c0\u67e5",
                "\u7a7f\u523a\u6d3b\u68c0",
                "\u57fa\u56e0\u68c0\u6d4b",
            ]
        )

        marrow = resolver.resolve(
            "\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09",
            candidate="\u767d\u8840\u75c5",
        )
        flow = resolver.resolve(
            "\u6d41\u5f0f\u7ec6\u80de\u672f\u514d\u75ab\u5206\u578b",
            candidate="\u767d\u8840\u75c5",
        )

        self.assertEqual(marrow.resolution_type, EQUIVALENT)
        self.assertEqual(
            marrow.resolved_exam,
            "\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09",
        )
        self.assertEqual(flow.resolution_type, EQUIVALENT)
        self.assertEqual(
            flow.resolved_exam,
            "\u6d41\u5f0f\u7ec6\u80de\u672f\u514d\u75ab\u5206\u578b",
        )
        self.assertNotEqual(marrow.resolved_exam, "\u7ec4\u7ec7\u75c5\u7406\u5b66\u68c0\u67e5")

    def test_pulmonary_vascular_special_requests_use_equivalent_catalog_exam(self):
        resolver = ExamResolver(
            catalog_names=[
                "\u80ba\u8840\u7ba1CTA",
                "\u8d85\u58f0\u5fc3\u52a8\u56fe\u53f3\u5fc3\u58f0\u5b66\u9020\u5f71",
                "\u80f8\u90e8\u589e\u5f3aCT",
                "\u80f8\u90e8CT\u626b\u63cf\uff08Chest CT\uff09",
            ]
        )

        cta = resolver.resolve("\u80ba\u52a8\u8109CTA")
        bubble_echo = resolver.resolve("\u53f3\u5fc3\u58f0\u5b66\u9020\u5f71")
        enhanced_ct = resolver.resolve("\u589e\u5f3a\u80f8\u90e8CT")

        self.assertEqual(cta.resolution_type, EQUIVALENT)
        self.assertEqual(cta.resolved_exam, "\u80ba\u8840\u7ba1CTA")
        self.assertEqual(bubble_echo.resolution_type, EQUIVALENT)
        self.assertEqual(
            bubble_echo.resolved_exam,
            "\u8d85\u58f0\u5fc3\u52a8\u56fe\u53f3\u5fc3\u58f0\u5b66\u9020\u5f71",
        )
        self.assertEqual(enhanced_ct.resolution_type, EQUIVALENT)
        self.assertEqual(enhanced_ct.resolved_exam, "\u80f8\u90e8\u589e\u5f3aCT")
        self.assertEqual(cta.diagnostic_coverage, 1.0)

    def test_unresolved_exam_stays_unresolved(self):
        resolver = ExamResolver(catalog_names=["血常规"])

        result = resolver.resolve("完全不存在的专科检查")

        self.assertEqual(result.resolution_type, UNRESOLVED)
        self.assertEqual(result.resolved_exam, "")


if __name__ == "__main__":
    unittest.main()
