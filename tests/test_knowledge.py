"""
The knowledge base loader.

The behaviour worth pinning down is the failure: an agent with no reference
material does not degrade into a worse agent, it degrades into a confident one.
"""

import pytest

from services import knowledge


def test_content_from_several_files_all_arrives(knowledge_dir):
    knowledge_dir("rates.md", "Two movers is $115 an hour.")
    knowledge_dir("faq.md", "We take TVs off the wall but do not mount them.")
    text = knowledge.all_context()
    assert "$115" in text
    assert "off the wall" in text


def test_the_readme_is_not_fed_to_the_model(knowledge_dir):
    """It is instructions for whoever edits the folder, not facts about us."""
    knowledge_dir("company.md", "We are open every day.")
    knowledge_dir("README.md", "Write facts here, not marketing fluff.")
    assert "marketing fluff" not in knowledge.all_context()


def test_filenames_are_shown_so_a_wrong_answer_can_be_traced_home(knowledge_dir):
    knowledge_dir("rates.md", "Two movers is $115 an hour.")
    assert "rates.md" in knowledge.all_context()


def test_an_empty_knowledge_base_is_fatal(tmp_path, monkeypatch):
    empty = tmp_path / "kb-empty"
    empty.mkdir()
    monkeypatch.setenv("KNOWLEDGE_DIR", str(empty))
    knowledge._load.cache_clear()
    with pytest.raises(knowledge.EmptyKnowledgeBase):
        knowledge.all_context()


def test_a_folder_of_blank_files_counts_as_empty(knowledge_dir):
    knowledge_dir("company.md", "   \n\n  ")
    with pytest.raises(knowledge.EmptyKnowledgeBase):
        knowledge.all_context()


def test_an_edit_is_picked_up_without_a_restart(knowledge_dir):
    """Someone fixes a rate in the file and the next customer gets the new one."""
    import time

    directory = knowledge_dir("rates.md", "Two movers is $115 an hour.")
    assert "$115" in knowledge.all_context()

    time.sleep(0.01)
    (directory / "rates.md").write_text("Two movers is $125 an hour.", encoding="utf-8")
    assert "$125" in knowledge.all_context()
