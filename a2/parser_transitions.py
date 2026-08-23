import sys

class PartialParse(object):
	def __init__(self, sentence):

		self.sentence = sentence

		self.stack = ["ROOT"]
		self.buffer = sentence[:]
		self.dependencies = []

	def parse_step(self, transition):
		if transition == "S":
			self.stack.append(self.buffer.pop(0))
		elif transition == "LA":
			head = self.stack[-1]
			dependent = self.stack.pop(-2)
			self.dependencies.append((head, dependent))
		elif transition == "RA":
			head = self.stack[-2]
			dependent = self.stack.pop(-1)
			self.dependencies.append((head, dependent))
		else:
			raise RuntimeError("unknown transition string")

	def parse(self, transitions):

		for transition in transitions:
			self.parse_step(transition)

		return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    partial_parses = [PartialParse(s) for s in sentences]
    unfinished = partial_parses[:]

    while unfinished:
        mini_batch = unfinished[:batch_size]
        transitions = model.predict(mini_batch)
        for parse, transition in zip(mini_batch, transitions):
            parse.parse_step(transition)
        unfinished = [p for p in unfinished
                      if not (len(p.stack) == 1 and not p.buffer)]

    return [p.dependencies for p in partial_parses]


def test_step(name, transition, stack, buf, deps,
			  ex_stack, ex_buf, ex_deps):
	"""Tests that a single parse step returns the expected output"""
	pp = PartialParse([])
	pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

	pp.parse_step(transition)
	stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
	assert stack == ex_stack, \
		"{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
	assert buf == ex_buf, \
		"{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
	assert deps == ex_deps, \
		"{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
	print("{:} test passed!".format(name))


def test_parse_step():
	"""Simple tests for the PartialParse.parse_step function
	Warning: these are not exhaustive
	"""
	test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
			  ("ROOT", "the", "cat"), ("sat",), ())
	test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
			  ("ROOT", "cat",), ("sat",), (("cat", "the"),))
	test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
			  ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
	"""Simple tests for the PartialParse.parse function
	Warning: these are not exhaustive
	"""
	sentence = ["parse", "this", "sentence"]
	dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
	dependencies = tuple(sorted(dependencies))
	expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
	assert dependencies == expected,  \
		"parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
	assert tuple(sentence) == ("parse", "this", "sentence"), \
		"parse test failed: the input sentence should not be modified"
	print("parse test passed!")


class DummyModel(object):
	"""Dummy model for testing the minibatch_parse function
	"""
	def __init__(self, mode = "unidirectional"):
		self.mode = mode

	def predict(self, partial_parses):
		if self.mode == "unidirectional":
			return self.unidirectional_predict(partial_parses)
		elif self.mode == "interleave":
			return self.interleave_predict(partial_parses)
		else:
			raise NotImplementedError()

	def unidirectional_predict(self, partial_parses):
		"""First shifts everything onto the stack and then does exclusively right arcs if the first word of
		the sentence is "right", "left" if otherwise.
		"""
		return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
				for pp in partial_parses]

	def interleave_predict(self, partial_parses):
		"""First shifts everything onto the stack and then interleaves "right" and "left".
		"""
		return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
				for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
	"""Tests the provided dependencies match the expected dependencies"""
	deps = tuple(sorted(deps))
	assert deps == ex_deps, \
		"{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
	"""Simple tests for the minibatch_parse function
	Warning: these are not exhaustive
	"""

	# Unidirectional arcs test
	sentences = [["right", "arcs", "only"],
				 ["right", "arcs", "only", "again"],
				 ["left", "arcs", "only"],
				 ["left", "arcs", "only", "again"]]
	deps = minibatch_parse(sentences, DummyModel(), 2)
	test_dependencies("minibatch_parse", deps[0],
					  (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
	test_dependencies("minibatch_parse", deps[1],
					  (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
	test_dependencies("minibatch_parse", deps[2],
					  (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
	test_dependencies("minibatch_parse", deps[3],
					  (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))

	# Out-of-bound test
	sentences = [["right"]]
	deps = minibatch_parse(sentences, DummyModel(), 2)
	test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))

	# Mixed arcs test
	sentences = [["this", "is", "interleaving", "dependency", "test"]]
	deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
	test_dependencies("minibatch_parse", deps[0],
					  (('ROOT', 'is'), ('dependency', 'interleaving'),
					  ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
	print("minibatch_parse test passed!")


if __name__ == '__main__':
	args = sys.argv
	if len(args) != 2:
		raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
	elif args[1] == "part_c":
		test_parse_step()
		test_parse()
	elif args[1] == "part_d":
		test_minibatch_parse()
	else:
		raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
