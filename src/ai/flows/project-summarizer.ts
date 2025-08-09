'use server';

/**
 * @fileOverview An AI agent that summarizes projects for different audiences.
 *
 * - summarizeProject - A function that handles the project summarization process.
 * - SummarizeProjectInput - The input type for the summarizeProject function.
 * - SummarizeProjectOutput - The return type for the summarizeProject function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SummarizeProjectInputSchema = z.object({
  projectDescription: z.string().describe('The detailed description of the project.'),
  audienceType: z
    .enum(['technical', 'non-technical'])
    .describe('The type of audience for the summary.'),
});
export type SummarizeProjectInput = z.infer<typeof SummarizeProjectInputSchema>;

const SummarizeProjectOutputSchema = z.object({
  summary: z.string().describe('The summarized description of the project.'),
});
export type SummarizeProjectOutput = z.infer<typeof SummarizeProjectOutputSchema>;

export async function summarizeProject(input: SummarizeProjectInput): Promise<SummarizeProjectOutput> {
  return summarizeProjectFlow(input);
}

const prompt = ai.definePrompt({
  name: 'summarizeProjectPrompt',
  input: {schema: SummarizeProjectInputSchema},
  output: {schema: SummarizeProjectOutputSchema},
  prompt: `You are an expert at summarizing complex projects for different audiences.

You will receive a project description and the intended audience.

Based on the audience type, adjust the summary to be appropriate for them. For technical audiences, use technical jargon and explain the implementation details. For non-technical audiences, explain the project in simple terms without getting into implementation details.

Project Description: {{{projectDescription}}}
Audience Type: {{{audienceType}}}

Summary:`,
});

const summarizeProjectFlow = ai.defineFlow(
  {
    name: 'summarizeProjectFlow',
    inputSchema: SummarizeProjectInputSchema,
    outputSchema: SummarizeProjectOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
