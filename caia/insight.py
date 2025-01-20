from docarray import BaseDoc
from docarray import DocList
import textwrap


# class FeatureInsight(BaseDoc):
#     feature: str
#     insight: str

class QuickInsight(BaseDoc):
    # dataset_title: str
    quick_insight: dict
class DeepInsight(BaseDoc):
    deep_insight: dict

# class DeepInsight(BaseDoc):
#     """
#     Generate a markdown report for deep insights on a dataset.

#     Returns:
#         str: The markdown report containing dataset overview, label insight, and feature insights.
#     """
#     dataset_title: str
#     dataset_description: str
#     overview: str
#     feature_insights: DocList[FeatureInsight]
#     label_insight: str

#     def generate_markdown_report(self) -> str:
#         deep_insight_markdown = textwrap.dedent(f"""
#         # {self.dataset_title} - Deep Insight Report

#         ## Overview
#         {self.overview}

#         ## Details

#         ### Label Insight
#         {self.label_insight}

#         """)
#         for feature_insight in self.feature_insights:
#             deep_insight_markdown += textwrap.dedent(f"""
#             ### {feature_insight.feature}

#             {feature_insight.insight}
#             """)
#         return deep_insight_markdown
