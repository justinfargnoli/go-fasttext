package fasttext

import (
	"os"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func build(t *testing.T) *FastText {
	ft := New(":memory:")

	file, err := os.Open("./testdata/wiki.en.vec")
	defer file.Close()
	if err != nil {
		t.Error(err)
	}

	err = ft.Build(file)
	if err != nil {
		t.Error(err)
	}

	return ft
}

func TestBuild(t *testing.T) {
	build(t)
}

func TestEmbeddingVector(t *testing.T) {
	ft := build(t)
	defer ft.Close()

	words := []string{"has", "but", "page", "#"}
	for _, word := range words {
		emb, err := ft.EmbeddingVector(word)
		if err != nil {
			t.Error(err)
		}
		t.Log(emb)
	}

	notExist := []string{"NotExist1", "Happiness"}
	for _, word := range notExist {
		_, err := ft.EmbeddingVector(word)
		if err != ErrNoEmbFound {
			t.Error("Should return not found")
		}
	}
}

func TestAllEmbeddingVectors(t *testing.T) {
	ft := build(t)
	defer ft.Close()

	if _, err := ft.AllEmbeddingVectors(); err != nil {
		t.Error(err)
	}
}

func TestMultiWordEmbeddingVector(t *testing.T) {
	ft := build(t)
	defer ft.Close()

	if _, err := ft.MultiWordEmbeddingVector([]string{"to", "be", "or", "not", "to", "be"}); err != nil {
		t.Error(err)
	}
}

func TestMostSimilarEmbeddingVector(t *testing.T) {
	ft := build(t)
	defer ft.Close()

	embeddingVector, err := ft.EmbeddingVector("has")
	if err != nil {
		t.Error(err)
	}

	if _, _, err := ft.MostSimilarEmbeddingVector(embeddingVector); err != nil {
		t.Error(err)
	}
}
